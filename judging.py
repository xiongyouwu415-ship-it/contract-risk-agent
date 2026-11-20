"""judging.py —— 判定 Agent（方案 6）。

六步循环状态机：
  1 加载任务快照 → 2 组装本轮输入 → 3 LLM 自主决策 → 4 工具分发执行
  → 5 回写快照、循环迭代 → 6 任务终止

关键约束：
  · Agent 是一套固定的 Prompt + 工具定义模板，并行跑的是 N 份任务快照，不是 N 套 Agent；
  · 快照里已有 FINISHED 结论时直接复用，不再调模型（防护三的简易版，也是重试幂等的基础）；
  · 轮次上限来自解析阶段打的复杂度标签，不是全局常量（方案 6.4 / 10.1）；
  · 单任务 token 超阈值立即熔断（方案 10.2 第三层）。
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
import db
import prompts
import tools
from llm import LLMClient, LLMError


class TokenBudgetExceeded(RuntimeError):
    """单任务 token 超过阈值，直接熔断（方案 10.2 第三层）。"""


@dataclass
class AgentOutcome:
    """判定/复核共用的任务产出。"""
    task_id: str
    status: str                                  # db.FINISHED / db.FAILED
    payload: Optional[Dict[str, Any]] = None     # 判定：风险结论；复核：最终结论
    rounds: int = 0
    tokens: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


# ==================================================================== JSON 解析
_FENCE = re.compile(r"^```(json)?|```$", re.M)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从模型回复里抠出第一个 JSON 对象。

    模型经常带 ```json 包裹、前后加解释、或者输出两个对象（先想后做），
    所以按「去围栏 → 整体解析 → 逐个花括号片段尝试」三级兜底，而不是简单的 json.loads。
    """
    if not text:
        return None
    cleaned = _FENCE.sub("", str(text)).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except (ValueError, TypeError):
        pass

    # 逐个候选花括号片段：从最外层 `{` 开始，配平深度
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except (ValueError, TypeError):
                    pass
                start = -1
    return None


def max_rounds_for(complexity: str) -> int:
    """复杂度 → 轮次上限。简单 3~4 轮，复杂 10~15 轮（方案 6.4）。"""
    return config.MAX_ROUNDS_COMPLEX if complexity == "complex" else config.MAX_ROUNDS_SIMPLE


# ==================================================================== 上下文拼装
def build_global_ctx_text(ctx: Dict[str, Any], term_index_limit: int = 30) -> str:
    """把全局共享上下文压成可注入的文本。只读，且必须控长度 —— 每轮都会带上它。"""
    lines: List[str] = []
    meta = ctx.get("meta") or {}
    if meta:
        lines.append("合同要素：" + "；".join(f"{k}={v}" for k, v in meta.items()))
    rules = ctx.get("global_rules") or []
    if rules:
        lines.append("全局规则：" + "；".join(rules[:4]))
    terms = ctx.get("custom_terms") or {}
    if terms:
        items = list(terms.items())[:5]
        lines.append("自定义术语：" + "；".join(f"{k}={v}" for k, v in items))
    index = ctx.get("term_index") or {}
    if index:
        items = list(index.items())[:term_index_limit]
        lines.append("条款索引：" + "；".join(f"{k}:{v}" for k, v in items))
    return "\n".join(lines) if lines else "（无全局上下文）"


def _inject_referenced_terms(text: str, contract_id: str,
                             already: set, terms_text: Dict[str, str]) -> List[str]:
    """按需注入：模型提到某个 term_id，就把那条条款原文追加进上下文（方案 6.2 第四步）。

    只认「本文合同内真实存在」的 term_id，防止模型编一个不存在的编号把上下文撑爆。
    """
    pattern = re.compile(rf"{re.escape(contract_id)}-T\d{{3}}")
    hits = [t for t in dict.fromkeys(pattern.findall(text or ""))
            if t not in already and t in terms_text]
    return hits


# ==================================================================== 主循环
async def run_judge(task_id: str, contract_id: str, term: Dict[str, Any],
                    global_ctx: Dict[str, Any], llm: LLMClient,
                    persist: bool = True) -> AgentOutcome:
    """判定一条条款。task_id 唯一，重入时优先复用已有快照的结论。"""
    term_id = term["term_id"]
    max_rounds = max_rounds_for(term.get("complexity", "simple"))

    # ---------- 第 1 步：加载任务快照 ----------
    snap = db.get_judge_snapshot(task_id) if persist else None
    if snap and snap["status"] == db.FINISHED and snap.get("risk_draft"):
        logging.info(f"[judge] {task_id} 命中已完成快照，直接复用结论")
        return AgentOutcome(task_id, db.FINISHED, snap["risk_draft"],
                            snap["cur_round"], snap["token_usage"], snap.get("tool_calls") or [])

    messages: List[Dict[str, Any]] = list(snap["messages"]) if snap else []
    cur_round = snap["cur_round"] if snap else 0
    tokens = snap["token_usage"] if snap else 0
    tool_calls: List[Dict[str, Any]] = list((snap.get("tool_calls") or []) if snap else [])

    if not messages:
        # ---------- 第 2 步：组装本轮输入（只在首轮做一次，后续靠历史消息续跑）----------
        messages = [
            {"role": "system", "content": prompts.JUDGE_SYSTEM.format(
                tool_list=tools.TOOL_LIST_FOR_PROMPT)},
            {"role": "user", "content": prompts.JUDGE_USER_TEMPLATE.format(
                global_ctx=build_global_ctx_text(global_ctx),
                term_id=term_id,
                complexity=term.get("complexity", "simple"),
                term_text=term["term_text"],
                extra="",
            )},
        ]

    injected: set = set()
    terms_text = {t["term_id"]: t["term_text"] for t in db.load_terms(contract_id)} if persist else {}

    # ---------- 第 3~6 步：循环推理 ----------
    while cur_round < max_rounds:
        cur_round += 1
        if tokens >= config.TOKEN_LIMIT_PER_TASK:
            err = f"token 已达上限 {config.TOKEN_LIMIT_PER_TASK}（累计 {tokens}），熔断"
            if persist:
                db.upsert_judge_snapshot(task_id, contract_id, term_id, db.FAILED,
                                         max_rounds, cur_round, messages,
                                         token_usage=tokens, tool_calls=tool_calls, last_error=err)
            return AgentOutcome(task_id, db.FAILED, None, cur_round, tokens, tool_calls, err)

        try:
            res = await llm.chat(messages)
        except LLMError as e:
            if persist:
                db.upsert_judge_snapshot(task_id, contract_id, term_id, db.FAILED,
                                         max_rounds, cur_round, messages,
                                         token_usage=tokens, tool_calls=tool_calls, last_error=str(e))
            return AgentOutcome(task_id, db.FAILED, None, cur_round, tokens, tool_calls, str(e))

        tokens += res.total_tokens
        messages.append({"role": "assistant", "content": res.text})

        obj = parse_json_object(res.text)
        if obj is None:
            # 格式漂移：回写一条纠偏提示，让模型下一轮自己改（方案风险表第一行）
            messages.append({"role": "user", "content":
                             "你的输出不是合法 JSON。请严格只输出一个 JSON 对象，不要任何解释。"})
            if persist:
                db.upsert_judge_snapshot(task_id, contract_id, term_id, db.WAIT_LLM,
                                         max_rounds, cur_round, messages,
                                         token_usage=tokens, tool_calls=tool_calls)
            continue

        action = str(obj.get("action", "")).lower()

        # ---------- 第 4 步：工具分发执行 ----------
        if action == "tool_call":
            tool_name = str(obj.get("tool", ""))
            args = obj.get("args") or {}
            try:
                result = tools.dispatch(tool_name, args)
                ok = bool(result.get("ok", True))
            except tools.ToolError as e:
                result = {"ok": False, "error": str(e)}
                ok = False
            tool_calls.append({"round": cur_round, "tool": tool_name, "args": args, "ok": ok})

            messages.append({"role": "user", "content": prompts.TOOL_RESULT_TEMPLATE.format(
                tool=tool_name, result=json.dumps(result, ensure_ascii=False))})

            # 按需注入：模型提到其他 term_id 就把原文追加进来，实现跨条款引用
            if persist and terms_text:
                for tid in _inject_referenced_terms(res.text, contract_id,
                                                    injected, terms_text):
                    injected.add(tid)
                    messages.append({"role": "user", "content":
                                     f"（按需注入）条款 {tid} 原文：\n{terms_text[tid]}"})

            # ---------- 第 5 步：回写快照，回到第 2 步继续 ----------
            if persist:
                db.upsert_judge_snapshot(task_id, contract_id, term_id, db.WAIT_LLM,
                                         max_rounds, cur_round, messages,
                                         token_usage=tokens, tool_calls=tool_calls)
            continue

        # ---------- 第 6 步：任务终止 ----------
        if action == "final":
            risk = _normalize_risk(obj)
            if persist:
                db.upsert_judge_snapshot(task_id, contract_id, term_id, db.FINISHED,
                                         max_rounds, cur_round, messages,
                                         risk_draft=risk, token_usage=tokens, tool_calls=tool_calls)
            return AgentOutcome(task_id, db.FINISHED, risk, cur_round, tokens, tool_calls)

        # 既不是 tool_call 也不是 final：当格式漂移处理，不下结论、不写状态
        messages.append({"role": "user", "content":
                         'action 只能是 "tool_call" 或 "final"，请重新输出。'})

    # 轮次耗尽
    err = f"超过最大轮次 {max_rounds} 仍未给出结论"
    if persist:
        db.upsert_judge_snapshot(task_id, contract_id, term_id, db.FAILED,
                                 max_rounds, cur_round, messages,
                                 token_usage=tokens, tool_calls=tool_calls, last_error=err)
    return AgentOutcome(task_id, db.FAILED, None, cur_round, tokens, tool_calls, err)


def _normalize_risk(obj: Dict[str, Any]) -> Dict[str, Any]:
    """把模型给的结论收敛成统一结构；缺字段就补默认，绝不让下游 KeyError。

    has_risk=false 的结论也要完整落库（作为「查过、无风险」的留痕），
    不能因为它没有风险就丢掉 —— 否则复核与统计都看不见这条条款被审过。
    """
    has_risk = bool(obj.get("has_risk", True))
    level = str(obj.get("risk_level", "") or "").strip()
    if not has_risk:
        level = "无"
    elif level not in ("高", "中", "低"):
        level = "中"                                  # 给了风险但等级不规范，按中危兜底
    return {
        "has_risk": has_risk,
        "risk_level": level,
        "reason": str(obj.get("reason", "") or "").strip(),
        "basis": str(obj.get("basis", "") or "").strip(),
        "evidence": str(obj.get("evidence", "") or "").strip(),
    }
