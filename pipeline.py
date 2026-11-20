"""pipeline.py —— 端到端编排：把调度器与四类节点串成一条链。

          解析Agent
              │  双路输出
     ┌────────┴─────────┐
  全局上下文          条款列表
     │                  │
     │            Fan-out 并行判定（N 份任务快照）
     │                  │
     │             证据校验（规则程序）
     │                  │ 三路分流（RouteTable）
     │        ┌─────────┼──────────┐
     │     定位成功   定位失败   缺少有效证据
     │        └────┬────┘          │
     └────────►复核Agent       判无效→回归池
                   │
              最终风险结论

编排原则：本文件只做「按表流转 + 收集结果」，不写业务判断。
哪条条款该进复核、哪条该进回归池，全由 scheduler.EVIDENCE_ROUTE 决定。
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import config
import db
import evidence as evidence_mod
import parsing
import reviewing
from judging import AgentOutcome, run_judge
from llm import LLMClient
from scheduler import EVIDENCE_ROUTE, Scheduler, summary

RISK_ORDER = {"高": 0, "中": 1, "低": 2, "无": 3}


def _judge_task_id(contract_id: str, term_id: str) -> str:
    return f"{contract_id}|judge|{term_id}"


def _review_task_id(contract_id: str, term_id: str) -> str:
    return f"{contract_id}|review|{term_id}"


async def review_contract(contract_id: str, file_name: str, raw_text: str,
                          llm: LLMClient, persist: bool = True,
                          resume: bool = True,
                          use_llm_parse: bool = True) -> Dict[str, Any]:
    """一份合同走完整条流水线，返回最终风险结论与验收指标。

    resume=True 时，若库里已有解析结果就直接复用，已 FINISHED 的判定/复核任务
    也不会重跑 —— 这就是「故障恢复仅重跑失败分支」的实现（方案 4.2-3）。
    """
    t0 = time.time()
    sched = Scheduler()
    if persist:
        db.init_db()

    # ------------------------------------------------ 1) 解析
    ctx = db.load_global_ctx(contract_id) if (persist and resume) else None
    terms = db.load_terms(contract_id) if (persist and resume) else []

    if ctx and terms:
        logging.info(f"[pipeline] 复用已解析结果：{len(terms)} 条条款")
        parsed = {"clean_text": ctx["clean_text"], "global_ctx": ctx, "terms": terms}
    else:
        parsed = await parsing.parse_contract(
            contract_id, file_name, raw_text, llm=llm,
            use_llm=use_llm_parse, persist=persist)

    clean_full_text = parsed["clean_text"]
    global_ctx = parsed["global_ctx"]
    terms = []
    for t in parsed["terms"]:
        t = dict(t)
        t["contract_id"] = contract_id
        terms.append(t)

    # ------------------------------------------------ 2) Fan-out 判定
    async def _judge(term: Dict[str, Any]) -> AgentOutcome:
        return await run_judge(_judge_task_id(contract_id, term["term_id"]), contract_id,
                               term, global_ctx, llm, persist=persist)

    judge_results = await sched.fan_out(terms, _judge, name_of=lambda t: t["term_id"])

    def _judge_done(term: Dict[str, Any]) -> bool:
        snap = db.get_judge_snapshot(_judge_task_id(contract_id, term["term_id"])) if persist else None
        return bool(snap and snap["status"] == db.FINISHED)

    judge_results = await sched.retry_failed(judge_results, terms, _judge,
                                             name_of=lambda t: t["term_id"],
                                             is_done=_judge_done)

    # ------------------------------------------------ 3) 证据校验 + 三路分流
    routed: List[Dict[str, Any]] = []          # 待复核
    finals: List[Dict[str, Any]] = []          # 直接定稿（无风险 / 判无效）
    term_by_id = {t["term_id"]: t for t in terms}

    for res in judge_results:
        term = term_by_id[res.key]
        outcome: Optional[AgentOutcome] = res.value if res.ok else None
        risk = (outcome.payload or {}) if outcome else {}

        if outcome is None or outcome.status != db.FINISHED:
            route = EVIDENCE_ROUTE.next("judge", db.FAILED)
            if persist:
                db.push_regression(contract_id, term["term_id"],
                                   f"判定任务失败：{(outcome.error if outcome else res.error) or '未知'}",
                                   {"term_no": term["term_no"]})
            finals.append({
                "term_id": term["term_id"], "term_no": term["term_no"],
                "action": "判定失败", "has_risk": False, "risk_level": "无",
                "reason": (outcome.error if outcome else res.error) or "判定任务失败",
                "evidence": "", "verify_status": "未校验", "similarity": 0.0,
                "needs_human": True, "route": route,
            })
            continue

        # 判定为「无风险」的条款没有证据片段可校验，直接定稿，不进复核也不进回归池
        if not risk.get("has_risk"):
            finals.append({
                "term_id": term["term_id"], "term_no": term["term_no"],
                "action": "无风险", "has_risk": False, "risk_level": "无",
                "reason": risk.get("reason", ""), "basis": risk.get("basis", ""),
                "evidence": "", "verify_status": "不适用", "similarity": 0.0,
                "anchor_start": None, "anchor_end": None,
                "needs_human": False, "route": "end",
            })
            continue

        # 规则化校验：相似度阈值分流，不做任何语义判断（方案 7）
        ev = evidence_mod.verify(clean_full_text, risk.get("evidence", ""))
        if persist:
            db.save_evidence(outcome.task_id, contract_id, term["term_id"],
                             ev["verify_status"], ev["start"], ev["end"],
                             ev["similarity"], ev["matched_text"])

        route = EVIDENCE_ROUTE.next("evidence", ev["verify_status"])
        if route == "regression":
            if persist:
                db.push_regression(contract_id, term["term_id"],
                                   "证据校验：缺少有效证据，结论判无效",
                                   {"evidence": risk.get("evidence", "")[:120],
                                    "similarity": ev["similarity"]})
            finals.append({
                "term_id": term["term_id"], "term_no": term["term_no"],
                "action": "判无效", "has_risk": False, "risk_level": "无",
                "reason": "引用的证据片段在原文中找不到，结论判无效",
                "evidence": risk.get("evidence", ""), "verify_status": ev["verify_status"],
                "similarity": ev["similarity"], "anchor_start": None, "anchor_end": None,
                "needs_human": True, "route": route,
            })
            continue

        routed.append({"term": term, "outcome": outcome, "evidence": ev, "route": route})

    # ------------------------------------------------ 4) 复核 Agent
    async def _review(item: Dict[str, Any]) -> reviewing.ReviewOutcome:
        term = item["term"]
        return await reviewing.run_review(
            _review_task_id(contract_id, term["term_id"]), item["outcome"], term,
            global_ctx, item["evidence"], llm, persist=persist)

    review_results = await sched.fan_out(routed, _review,
                                         name_of=lambda x: x["term"]["term_id"])

    def _review_done(item: Dict[str, Any]) -> bool:
        snap = db.get_review_snapshot(
            _review_task_id(contract_id, item["term"]["term_id"])) if persist else None
        return bool(snap and snap["status"] == db.FINISHED)

    review_results = await sched.retry_failed(review_results, routed, _review,
                                              name_of=lambda x: x["term"]["term_id"],
                                              is_done=_review_done)

    routed_by_id = {x["term"]["term_id"]: x for x in routed}
    for res in review_results:
        item = routed_by_id.get(res.key)
        if item is None:
            continue
        if res.ok and res.value is not None:
            final = dict(res.value.payload or {})
            final["term_no"] = item["term"]["term_no"]
            final["route"] = item["route"]
            finals.append(final)
        else:
            # 复核分支整体失败：退回判定原文，标记待人工确认，绝不静默丢结论
            risk = (item["outcome"].payload or {})
            finals.append({
                "term_id": res.key, "term_no": item["term"]["term_no"],
                "action": "复核失败", "has_risk": True,
                "risk_level": risk.get("risk_level", ""),
                "reason": risk.get("reason", ""), "evidence": risk.get("evidence", ""),
                "verify_status": item["evidence"]["verify_status"],
                "similarity": item["evidence"]["similarity"],
                "needs_human": True, "route": item["route"], "error": res.error,
            })

    # ------------------------------------------------ 5) 收尾
    finals.sort(key=lambda x: (RISK_ORDER.get(str(x.get("risk_level")), 9),
                               x.get("term_no", 0)))
    stats = db.contract_stats(contract_id) if persist else {}
    duration = round(time.time() - t0, 2)

    risks = [f for f in finals if f.get("has_risk")]
    logging.info(f"[pipeline] {file_name} 完成：条款 {len(terms)}，"
                 f"风险 {len(risks)}，待人工 {sum(1 for f in finals if f.get('needs_human'))}，"
                 f"耗时 {duration}s")

    return {
        "contract_id": contract_id,
        "file_name": file_name,
        "terms_total": len(terms),
        "risk_total": len(risks),
        "needs_human": sum(1 for f in finals if f.get("needs_human")),
        "final": finals,
        "stats": stats,
        "scheduler": {**summary(judge_results), "llm_calls": getattr(llm, "calls", 0)},
        "duration_s": duration,
    }


def run_contract_sync(*args, **kwargs) -> Dict[str, Any]:
    """同步包装，方便脚本里直接调用。"""
    return asyncio.run(review_contract(*args, **kwargs))
