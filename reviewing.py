"""reviewing.py —— 复核 Agent（方案 8）。

输入四件套：判定结论 + 证据锚点片段 + 全局上下文 + 本条原始条款。
内部循环与判定 Agent 同构（读快照 → 组装 → LLM 决策 → 工具分发 → 回写），
区别只在动作空间：采信 / 修正 / 补充核验 / 驳回。

与证据校验的分工（方案 8.4）：证据校验解决「证据是否真实存在」，
复核解决「这个判断本身对不对」。前者是脚本，后者是 Agent。
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config
import db
import prompts
import tools
from judging import AgentOutcome, parse_json_object, build_global_ctx_text
from llm import LLMClient, LLMError

# 复核动作（方案 8.3 表 6）
ACTION_ACCEPT = "采信"
ACTION_REVISE = "修正"
ACTION_REJECT = "驳回"
ACTION_VERIFY = "补充核验"

REVIEW_MAX_ROUNDS = 6           # 复核比判定轻，轮次上限压小，控成本


@dataclass
class ReviewOutcome(AgentOutcome):
    """复核产出：action + 最终风险结论。"""
    action: str = ""
    needs_human: bool = False


async def run_review(task_id: str, judge_outcome: AgentOutcome, term: Dict[str, Any],
                     global_ctx: Dict[str, Any], evidence: Dict[str, Any],
                     llm: LLMClient, persist: bool = True) -> ReviewOutcome:
    """复核一条判定结论。task_id 为复核任务自己的 id（与判定任务分开存）。"""
    contract_id = term["contract_id"] if "contract_id" in term else ""
    term_id = term["term_id"]

    snap = db.get_review_snapshot(task_id) if persist else None
    if snap and snap["status"] == db.FINISHED and snap.get("final_risk"):
        logging.info(f"[review] {task_id} 命中已完成快照，直接复用复核结论")
        return ReviewOutcome(task_id, db.FINISHED, snap["final_risk"], snap["cur_round"],
                             snap["token_usage"], action=snap.get("action") or "",
                             needs_human=bool((snap["final_risk"] or {}).get("needs_human")))

    messages: List[Dict[str, Any]] = list(snap["messages"]) if snap else []
    cur_round = snap["cur_round"] if snap else 0
    tokens = snap["token_usage"] if snap else 0

    verify_status = evidence.get("verify_status", db.EVIDENCE_NONE)
    similarity = evidence.get("similarity", 0.0) or 0.0
    anchor_text = evidence.get("matched_text") or ""

    if not messages:
        extra = ""
        if verify_status == db.EVIDENCE_FAIL:
            extra = ("注意：证据校验未能在原文中稳定定位该片段（相似度低于阈值），"
                     "这条结论属于降级结论，请更严格地判断是否成立。\n")
        messages = [
            {"role": "system", "content": prompts.REVIEW_SYSTEM.format(
                tool_list=tools.TOOL_LIST_FOR_PROMPT)},
            {"role": "user", "content": prompts.REVIEW_USER_TEMPLATE.format(
                global_ctx=build_global_ctx_text(global_ctx),
                term_id=term_id,
                term_text=term["term_text"],
                judge_result=_judge_brief(judge_outcome.payload or {}),
                verify_status=verify_status,
                anchor_text=anchor_text or "（无）",
                similarity=f"{similarity:.2f}",
                extra=extra,
            )},
        ]

    while cur_round < REVIEW_MAX_ROUNDS:
        cur_round += 1
        if tokens >= config.TOKEN_LIMIT_PER_TASK:
            err = f"token 已达上限 {config.TOKEN_LIMIT_PER_TASK}（累计 {tokens}），熔断"
            return _fail(task_id, contract_id, term_id, judge_outcome, evidence,
                         messages, cur_round, tokens, err, persist)

        try:
            res = await llm.chat(messages)
        except LLMError as e:
            return _fail(task_id, contract_id, term_id, judge_outcome, evidence,
                         messages, cur_round, tokens, str(e), persist)

        tokens += res.total_tokens
        messages.append({"role": "assistant", "content": res.text})
        obj = parse_json_object(res.text)

        if obj is None:
            messages.append({"role": "user", "content":
                             "你的输出不是合法 JSON。请严格只输出一个 JSON 对象。"})
            continue

        action = str(obj.get("action", "")).lower()

        if action == "tool_call":
            tool_name = str(obj.get("tool", ""))
            args = obj.get("args") or {}
            try:
                result = tools.dispatch(tool_name, args)
            except tools.ToolError as e:
                result = {"ok": False, "error": str(e)}
            messages.append({"role": "user", "content": prompts.TOOL_RESULT_TEMPLATE.format(
                tool=tool_name, result=str(result))})
            if persist:
                db.upsert_review_snapshot(task_id, judge_outcome.task_id, contract_id, term_id,
                                          db.WAIT_LLM, REVIEW_MAX_ROUNDS, cur_round,
                                          messages, token_usage=tokens)
            continue

        final = _build_final(obj, judge_outcome.payload or {}, evidence, term_id)
        if final is None:
            messages.append({"role": "user", "content":
                             'action 只能是 "accept" / "revise" / "reject" / "tool_call"，请重新输出。'})
            continue

        if persist:
            db.upsert_review_snapshot(task_id, judge_outcome.task_id, contract_id, term_id,
                                      db.FINISHED, REVIEW_MAX_ROUNDS, cur_round, messages,
                                      action=final["action"], final_risk=final,
                                      token_usage=tokens)
        return ReviewOutcome(task_id, db.FINISHED, final, cur_round, tokens,
                             action=final["action"], needs_human=final["needs_human"])

    err = f"超过最大轮次 {REVIEW_MAX_ROUNDS} 仍未给出处置决定"
    return _fail(task_id, contract_id, term_id, judge_outcome, evidence,
                 messages, cur_round, tokens, err, persist)


# ---------------------------------------------------------------- 内部工具
def _judge_brief(risk: Dict[str, Any]) -> str:
    if not risk:
        return "（无判定结论）"
    return (f"是否有风险：{'是' if risk.get('has_risk') else '否'}；"
            f"等级：{risk.get('risk_level', '')}；理由：{risk.get('reason', '')}；"
            f"依据：{risk.get('basis', '')}；引用原文：{risk.get('evidence', '')}")


def _build_final(obj: Dict[str, Any], judge_risk: Dict[str, Any],
                 evidence: Dict[str, Any], term_id: str) -> Optional[Dict[str, Any]]:
    """把复核动作收敛成最终风险结论。返回 None 表示动作不合法，让模型重来。"""
    action = str(obj.get("action", "")).lower()
    verify_status = evidence.get("verify_status", db.EVIDENCE_NONE)
    needs_human = verify_status == db.EVIDENCE_FAIL      # 定位失败 → 降级待人工确认

    if action == "accept":
        return {
            "term_id": term_id,
            "action": ACTION_ACCEPT,
            "has_risk": bool(judge_risk.get("has_risk", True)),
            "risk_level": judge_risk.get("risk_level", ""),
            "reason": judge_risk.get("reason", ""),
            "basis": judge_risk.get("basis", ""),
            "evidence": judge_risk.get("evidence", ""),
            "anchor_start": evidence.get("start"),
            "anchor_end": evidence.get("end"),
            "verify_status": verify_status,
            "similarity": evidence.get("similarity", 0.0),
            "needs_human": needs_human,
        }

    if action == "revise":
        level = str(obj.get("risk_level", "") or "").strip()
        if level not in ("高", "中", "低"):
            level = judge_risk.get("risk_level", "中")
        return {
            "term_id": term_id,
            "action": ACTION_REVISE,
            "has_risk": True,
            "risk_level": level,
            "reason": str(obj.get("reason", "") or judge_risk.get("reason", "")).strip(),
            "basis": str(obj.get("basis", "") or judge_risk.get("basis", "")).strip(),
            "evidence": str(obj.get("evidence", "") or judge_risk.get("evidence", "")).strip(),
            "anchor_start": evidence.get("start"),
            "anchor_end": evidence.get("end"),
            "verify_status": verify_status,
            "similarity": evidence.get("similarity", 0.0),
            "needs_human": needs_human,
        }

    if action == "reject":
        # 驳回：证据找到了，但这段原文不构成风险 —— 清除该条风险标记（方案 8.3）
        return {
            "term_id": term_id,
            "action": ACTION_REJECT,
            "has_risk": False,
            "risk_level": "无",
            "reason": str(obj.get("reason", "") or "复核判定为误报").strip(),
            "basis": "",
            "evidence": "",
            "anchor_start": None,
            "anchor_end": None,
            "verify_status": verify_status,
            "similarity": evidence.get("similarity", 0.0),
            "needs_human": False,
        }

    return None


def _fail(task_id: str, contract_id: str, term_id: str, judge_outcome: AgentOutcome,
          evidence: Dict[str, Any], messages: List[Dict[str, Any]],
          cur_round: int, tokens: int, err: str, persist: bool) -> ReviewOutcome:
    """复核失败不能丢掉判定结论 —— 退回判定原文，标记待人工确认。"""
    logging.error(f"[review] {task_id} 复核失败：{err}")
    fallback = {
        "term_id": term_id,
        "action": "复核失败",
        "has_risk": bool((judge_outcome.payload or {}).get("has_risk", True)),
        "risk_level": (judge_outcome.payload or {}).get("risk_level", ""),
        "reason": (judge_outcome.payload or {}).get("reason", ""),
        "basis": (judge_outcome.payload or {}).get("basis", ""),
        "evidence": (judge_outcome.payload or {}).get("evidence", ""),
        "anchor_start": evidence.get("start"),
        "anchor_end": evidence.get("end"),
        "verify_status": evidence.get("verify_status", db.EVIDENCE_NONE),
        "similarity": evidence.get("similarity", 0.0),
        "needs_human": True,
    }
    if persist:
        db.upsert_review_snapshot(task_id, judge_outcome.task_id, contract_id, term_id,
                                  db.FAILED, REVIEW_MAX_ROUNDS, cur_round, messages,
                                  action="复核失败", final_risk=fallback,
                                  token_usage=tokens, last_error=err)
    return ReviewOutcome(task_id, db.FAILED, fallback, cur_round, tokens,
                         action="复核失败", needs_human=True, error=err)
