"""tools.py —— 自研 Function Calling 工具层（方案 6.3）。

无框架封装：工具注册表 + JSON 协议 + 分发器。判定 Agent 与复核 Agent 共用同一套。
分发器只做三件事：白名单校验、参数校验、执行；未注册的工具直接拒绝，
并把错误提示回写给模型，由模型在下一轮自行纠正（不自作主张替它猜）。

三类工具：
  search_law     法条检索
  calc_amount    金额核算
  check_redline  红线规则查询

数据来源：data/laws.json。方案 1.2 明确「法条数据、红线规则库的治理不在本方案范围，
仅约定工具接口」，所以这里实现的是接口 + 样例数据，真实数据由上游灌入即可，
Agent 侧代码不用动。
"""
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

import config


# ==================================================================== 数据加载
def _load_law_db() -> Dict[str, Any]:
    if not os.path.exists(config.LAW_DB_PATH):
        return {"laws": [], "redlines": []}
    try:
        with open(config.LAW_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"laws": [], "redlines": []}


def _bigrams(text: str) -> set:
    t = re.sub(r"\s+", "", text or "")
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) > 1 else {t} if t else set()


def _overlap_score(query: str, doc: str) -> float:
    """字符 2-gram 覆盖率：纯标准库，中文不用分词也不至于太差。"""
    q, d = _bigrams(query), _bigrams(doc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


# ==================================================================== 工具实现
def search_law(query: str, top_k: int = 3, **_) -> Dict[str, Any]:
    """按法条关键词或条文引用检索法条原文。"""
    db = _load_law_db()
    laws = db.get("laws", [])
    if not laws:
        return {"ok": False, "error": "法条库为空（data/laws.json 未接入），请勿据此下结论"}

    scored = []
    for item in laws:
        haystack = f"{item.get('law_name', '')}{item.get('article_no', '')}{item.get('text', '')}"
        s = _overlap_score(query, haystack)
        # 条文号直接命中给一次加分，避免「第十条」这类短查询被长条文淹没
        if item.get("article_no") and item["article_no"] in (query or ""):
            s += 0.5
        if s > 0:
            scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)

    hits = [{
        "law_name": it.get("law_name", ""),
        "article_no": it.get("article_no", ""),
        "text": it.get("text", ""),
        "effective": it.get("effective", ""),
        "version": it.get("version", ""),
        "match": round(s, 3),
    } for s, it in scored[:max(1, int(top_k))]]

    if not hits:
        return {"ok": True, "hits": [], "note": "未检索到相关法条，请勿凭空引用法律依据"}
    return {"ok": True, "hits": hits}


def calc_amount(base_amount: float = 0.0, rate: float = 0.0,
                rate_type: str = "total", periods: float = 1.0,
                agreed_cap: Optional[float] = None, **_) -> Dict[str, Any]:
    """金额核算：违约金/赔偿金的金额、比例、是否超上限。

    rate_type: total（一次性比例）/ day / month / year（按周期累计）
    agreed_cap: 合同约定的上限金额，用于判断是否超出约定
    """
    try:
        base = float(base_amount)
        r = float(rate)
        p = float(periods)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"金额参数不是合法数字：{e}"}

    if base < 0 or r < 0:
        return {"ok": False, "error": "金额与比例必须为非负数"}

    multiplier = {"total": 1.0, "day": p, "month": p, "year": p}.get(rate_type, 1.0)
    amount = round(base * r * multiplier, 2)
    pct_of_base = round(amount / base * 100, 2) if base else None

    notes: List[str] = []
    # 与《民法典》第 585 条口径对齐：约定违约金过分高于损失的可请求调减（司法惯例约 130%）
    if pct_of_base is not None and pct_of_base > 130:
        notes.append(f"违约金约为基数的 {pct_of_base}%，超出「损失 130%」的司法调减惯例，存在被调减风险")
    if agreed_cap is not None:
        try:
            cap = float(agreed_cap)
            if amount > cap:
                notes.append(f"核算金额 {amount} 已超出合同约定上限 {cap}")
        except (TypeError, ValueError):
            notes.append("agreed_cap 不是合法数字，未做上限比对")

    return {
        "ok": True,
        "base_amount": base,
        "rate": r,
        "rate_type": rate_type,
        "periods": p,
        "amount": amount,
        "pct_of_base": pct_of_base,
        "notes": notes,
    }


def check_redline(text: str = "", **_) -> Dict[str, Any]:
    """查询条款是否命中红线规则库。"""
    db = _load_law_db()
    rules = db.get("redlines", [])
    if not rules:
        return {"ok": False, "error": "红线规则库为空（data/laws.json 未接入）"}

    hits = []
    for rule in rules:
        keys = rule.get("keywords", [])
        if not keys:
            continue
        hit_keys = [k for k in keys if k and k in (text or "")]
        if hit_keys:
            hits.append({
                "rule_id": rule.get("rule_id", ""),
                "rule": rule.get("rule", ""),
                "risk_level": rule.get("risk_level", ""),
                "advice": rule.get("advice", ""),
                "hit_keywords": hit_keys,
            })
    return {"ok": True, "hits": hits,
            "note": "未命中红线规则" if not hits else f"命中 {len(hits)} 条红线规则"}


# ==================================================================== 注册表与分发器
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "search_law": {
        "func": search_law,
        "required": ["query"],
        "desc": "search_law(法条检索)：检索与条款相关的法律法规条文。入参 query=法条关键词或条文编号。",
    },
    "calc_amount": {
        "func": calc_amount,
        "required": ["base_amount", "rate"],
        "desc": "calc_amount(金额核算)：核算违约金/赔偿金。入参 base_amount=基数, rate=比例, "
                "rate_type=total/day/month/year, periods=周期数, agreed_cap=约定上限。",
    },
    "check_redline": {
        "func": check_redline,
        "required": ["text"],
        "desc": "check_redline(红线规则查询)：查询条款要素是否命中红线规则。入参 text=待查条款文本。",
    },
}

# 给 Prompt 用的工具清单（模型看到的就是这份）
TOOL_LIST_FOR_PROMPT = "\n".join(f"- {v['desc']}" for v in TOOL_REGISTRY.values())


class ToolError(Exception):
    """工具调用被拒绝或执行失败，错误信息会回写给模型让它自己纠正。"""


def dispatch(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """白名单校验 → 参数校验 → 执行。任何异常都转成模型可读的错误。"""
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ToolError(
            f"未注册的工具 {tool_name!r}，可用工具只有：{list(TOOL_REGISTRY)}"
        )
    if not isinstance(args, dict):
        raise ToolError("args 必须是 JSON 对象")

    missing = [k for k in spec["required"] if k not in args]
    if missing:
        raise ToolError(f"工具 {tool_name} 缺少必填参数：{missing}")

    func: Callable[..., Dict[str, Any]] = spec["func"]
    try:
        return func(**args)
    except TypeError as e:
        raise ToolError(f"工具 {tool_name} 参数不匹配：{e}")
    except Exception as e:                                   # 工具内部异常不能炸掉 Agent 循环
        raise ToolError(f"工具 {tool_name} 执行异常：{type(e).__name__}: {e}")
