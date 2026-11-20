"""metrics.py —— 验收指标（方案 12.3）与金标回归。

方案 12.3 列了五项观测项，这里逐条落地：
  1. 证据校验定位成功率   —— 定位成功 / (定位成功 + 定位失败)，缺证单列
  2. 复核修正与驳回比例   —— 评估判定 Agent 稳定性
  3. 单合同平均 token 消耗 与 端到端处理时长
  4. 故障恢复后重跑范围   —— 用「命中已完成快照的分支数」体现
  5. 金标比对（补充项）   —— 与 final_gt 的人工标注做字符区间比对，
                             输出每条样本的 IoU / Precision / Recall 与判定档位

判定档位（仅评测层，不影响运行时复核动作）：
  match      IoU ≥ 0.8              定位成功
  ambiguous  0.3 ≤ IoU < 0.8        匹配模糊，待人工确认（沿用原 ≥0.3 命中口径）
  too_broad  IoU < 0.3 但 Recall 高、Precision 低   证据范围过宽，待人工确认，
                                                    不计为证据缺失
  conflict   标注未被认领，但有【已被认领的 pred】同时覆盖它
             —— 一条证据盖住了两条标注，贪心一对一只能认领一条；
                这是匹配策略的牺牲品，不算模型漏检，单独计数待人工裁定
  miss       其余                    漏检
优先级：IoU 两档先判，too_broad 只接管 IoU 低于下限的那一档，conflict 只发生在
未被认领的标注上；阈值全部可配置。
"""
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import config
import db
from evidence import find_span

# 金标换算到 clean 坐标系时允许的最低相似度（find_span 归一化匹配），
# 低于此值认为该标注在 clean 文本里找不到对应片段，退回 raw 坐标并打标记
GT_MAP_SIMILARITY = 0.9

IOU_THRESHOLD = config.GOLDEN_IOU_MIN   # 兼容旧引用：命中下限

VERDICT_LABEL = {
    "match": "定位成功（自动采信档）",
    "ambiguous": "匹配模糊（待人工确认）",
    "too_broad": "证据范围过宽（待人工确认）",
    "conflict": "匹配冲突（证据被其他标注占用，待人工裁定）",
    "miss": "漏检",
}


# ==================================================================== 区间工具
def _norm_span(span: Any) -> Optional[tuple]:
    """把任意输入收敛成合法字符区间；空 / None / 倒置 / 负偏移一律视为无效。"""
    if not span:
        return None
    try:
        s, e = int(span[0]), int(span[1])
    except (TypeError, ValueError, IndexError):
        return None
    if e <= s or s < 0:            # 空区间、倒置、越界都不参与计算
        return None
    return (s, e)


def overlap_metrics(pred_span: Any, gt_span: Any) -> Dict[str, Any]:
    """基于字符区间交集计算三指标，空区间/越界返回全 0，不抛除零错误。

      Precision = 交集 / 预测区间长度   —— 引用范围有多精准
      Recall    = 交集 / GT 区间长度    —— GT 被覆盖了多少
      IoU       = 交集 / 并集           —— 整体定位质量
    """
    out = {"iou": 0.0, "precision": 0.0, "recall": 0.0,
           "inter": 0, "pred_len": 0, "gt_len": 0}
    p, g = _norm_span(pred_span), _norm_span(gt_span)
    if p is None or g is None:
        return out
    out["pred_len"] = p[1] - p[0]
    out["gt_len"] = g[1] - g[0]
    inter = max(0, min(p[1], g[1]) - max(p[0], g[0]))
    out["inter"] = inter
    if inter == 0:
        return out
    out["precision"] = round(inter / out["pred_len"], 4)
    out["recall"] = round(inter / out["gt_len"], 4)
    union = max(p[1], g[1]) - min(p[0], g[0])
    out["iou"] = round(inter / union, 4) if union > 0 else 0.0
    return out


def iou(a: Optional[tuple], b: Optional[tuple]) -> float:
    """兼容旧接口：只取 IoU 一项。"""
    return overlap_metrics(a, b)["iou"]


def classify_pair(m: Dict[str, Any], iou_strong: Optional[float] = None,
                  iou_min: Optional[float] = None,
                  broad_recall: Optional[float] = None,
                  broad_precision: Optional[float] = None) -> str:
    """按三档 + 过宽识别给出档位。阈值不传则用 config 里的可配置默认值。"""
    iou_strong = config.GOLDEN_IOU_STRONG if iou_strong is None else iou_strong
    iou_min = config.GOLDEN_IOU_MIN if iou_min is None else iou_min
    broad_recall = config.TOO_BROAD_RECALL if broad_recall is None else broad_recall
    broad_precision = (config.TOO_BROAD_PRECISION if broad_precision is None
                       else broad_precision)
    v = m["iou"]
    if v >= iou_strong:
        return "match"
    if v >= iou_min:
        return "ambiguous"
    # 只在原「漏检档」里接管范围过宽：GT 几乎被完整覆盖，只是引用太宽
    if m["recall"] >= broad_recall and m["precision"] <= broad_precision:
        return "too_broad"
    return "miss"


def load_golden(file_name: str) -> List[Dict[str, Any]]:
    """读人工标注：final_gt/<合同名>.json 里的 ground_truth 列表。"""
    path = os.path.join(config.GOLDEN_DIR, os.path.splitext(file_name)[0] + ".json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return data.get("ground_truth", [])


def _pred_span(item: Dict[str, Any], clean_text: str) -> Optional[tuple]:
    """pred 的锚点：优先用证据校验算出来的偏移，没有再按 evidence 文本回查一次。"""
    s, e = item.get("anchor_start"), item.get("anchor_end")
    if s is not None and e is not None and e > s:
        return (int(s), int(e))
    text = (item.get("evidence") or "").strip()
    if text and clean_text:
        pos = clean_text.find(text)
        if pos != -1:
            return (pos, pos + len(text))
    return None


# ==================================================================== 金标比对
def _map_golden_spans(golden: List[Dict[str, Any]],
                      clean_text: str) -> List[Dict[str, Any]]:
    """把金标区间从【原文 raw 坐标】换算到【clean_text 坐标】。

    背景：final_gt 的 start_idx/end_idx 是对原文标注的，而流水线的条款与证据
    锚点全部是 clean_text（parsing.clean_text 行级清洗后）坐标。两套坐标混比
    会把命中的证据错判成漏检。这里复用 evidence.find_span 的归一化匹配，按
    标注文本在 clean_text 里重新定位；找不到的退回 raw 坐标并在样本里打标记。
    只影响评测，不改任何运行时偏移逻辑。
    """
    mapped = []
    for g in golden:
        raw_span = (g.get("start_idx"), g.get("end_idx"))
        span, ok, sim = raw_span, False, 0.0
        text = (g.get("text") or "").strip()
        if text and clean_text:
            hit = find_span(clean_text, text)
            sim = float(hit.get("similarity") or 0.0)
            if hit.get("start") is not None and sim >= GT_MAP_SIMILARITY:
                span, ok = (int(hit["start"]), int(hit["end"])), True
        mapped.append({"span": span, "mapped": ok, "similarity": round(sim, 4)})
    return mapped


def score_against_golden(result: Dict[str, Any], clean_text: str,
                         iou_threshold: Optional[float] = None) -> Dict[str, Any]:
    """最终风险结论 vs 人工标注：召回 / 误报 / 高危漏检 / 每条样本明细。

    匹配策略沿用原有实现：同一份合同内、每条 pred 取剩余标注中 IoU 最高者
    （贪心一对一），每条标注只被认领一次。认领条件从「IoU ≥ 下限」放宽为
    「档位非 miss」，即范围过宽（too_broad）也会认领，但单独计数，不计入命中。
    """
    if iou_threshold is None:
        iou_threshold = config.GOLDEN_IOU_MIN
    golden = load_golden(result.get("file_name", ""))
    preds = [p for p in result.get("final", []) if p.get("has_risk")]

    # 金标 raw 坐标 → clean 坐标（找不到对应片段的退回 raw 坐标）
    gt_maps = _map_golden_spans(golden, clean_text)

    # 预计算全部 pred×GT 的三指标矩阵（规模很小），供匹配与明细输出共用
    pairs = [[overlap_metrics(_pred_span(p, clean_text), gm["span"])
              for gm in gt_maps] for p in preds]

    gt_matched = [False] * len(golden)
    pred_hit = [False] * len(preds)
    claimed: Dict[int, Dict[str, Any]] = {}

    for pi, _ in enumerate(preds):
        best_iou, best_gid = 0.0, None
        for gid in range(len(golden)):
            if gt_matched[gid]:
                continue
            if pairs[pi][gid]["iou"] > best_iou:
                best_iou, best_gid = pairs[pi][gid]["iou"], gid
        if best_gid is None:
            continue
        verdict = classify_pair(pairs[pi][best_gid], iou_min=iou_threshold)
        if verdict == "miss":
            continue
        gt_matched[best_gid] = True
        pred_hit[pi] = True
        claimed[best_gid] = {"pred_index": pi, "metrics": pairs[pi][best_gid],
                             "verdict": verdict}

    # 每条标注产出一条样本明细；未认领的标 miss，附上与其 IoU 最高的 pred 作对照
    samples = []
    by_level = defaultdict(lambda: {"hit": 0, "miss": 0, "broad": 0, "conflict": 0})
    for gid, g in enumerate(golden):
        gm = gt_maps[gid]
        gs = gm["span"]
        if gid in claimed:
            c = claimed[gid]
            m, verdict, pi = c["metrics"], c["verdict"], c["pred_index"]
        else:
            # 未被认领：先查是否有【已被认领的 pred】同时也覆盖了这条标注 ——
            # 一条证据盖住两条标注时，贪心一对一只能认领一条，另一条记 conflict
            # 而不是漏检；没有覆盖者才真正计 miss
            cover_pi, cover_m = None, None
            for pi2 in range(len(preds)):
                if not pred_hit[pi2]:
                    continue
                m2 = pairs[pi2][gid]
                if classify_pair(m2, iou_min=iou_threshold) == "miss":
                    continue
                if cover_pi is None or m2["iou"] > cover_m["iou"]:
                    cover_pi, cover_m = pi2, m2
            if cover_pi is not None:
                verdict, pi, m = "conflict", cover_pi, cover_m
            else:
                # 诊断用：取与该标注 IoU 最高的 pred（无论是否已被别的标注认领）
                best_iou, best_pi = 0.0, None
                for pi2 in range(len(preds)):
                    if pairs[pi2][gid]["iou"] > best_iou:
                        best_iou, best_pi = pairs[pi2][gid]["iou"], pi2
                m = pairs[best_pi][gid] if best_pi is not None else overlap_metrics(None, gs)
                verdict, pi = "miss", best_pi
        lv = str(g.get("risk_level", "")) or "其他"
        raw_s = (g.get("start_idx"), g.get("end_idx"))
        samples.append({
            "gt_index": gid, "risk_level": lv,
            "gt_span": [gs[0], gs[1]],
            "gt_span_raw": [raw_s[0], raw_s[1]],
            "gt_mapped": gm["mapped"], "gt_map_similarity": gm["similarity"],
            "gt_len": (gs[1] - gs[0]) if gs[0] is not None and gs[1] is not None else 0,
            "gt_text": str(g.get("text", ""))[:40],
            "pred_index": pi, "pred_span": list(_pred_span(preds[pi], clean_text) or ()) if pi is not None else None,
            "inter": m["inter"], "pred_len": m["pred_len"],
            "iou": m["iou"], "precision": m["precision"], "recall": m["recall"],
            "verdict": verdict, "verdict_label": VERDICT_LABEL[verdict],
        })
        if verdict == "miss":
            by_level[lv]["miss"] += 1
        elif verdict == "conflict":
            by_level[lv]["conflict"] += 1
        else:
            by_level[lv]["hit" if verdict != "too_broad" else "broad"] += 1

    verdict_counts = defaultdict(int)
    for s in samples:
        verdict_counts[s["verdict"]] += 1
    hit = verdict_counts["match"] + verdict_counts["ambiguous"]
    broad = verdict_counts["too_broad"]
    conflict = verdict_counts["conflict"]
    total = len(golden)
    claimed_preds = sum(1 for h in pred_hit if h)
    high_risk_miss = sum(1 for s in samples
                         if s["verdict"] == "miss" and s["risk_level"] == "高")

    return {
        "golden_total": total,
        "golden_hit": hit,
        "too_broad": broad,
        "recall": round(hit / total, 4) if total else 0.0,
        "recall_with_broad": round((hit + broad) / total, 4) if total else 0.0,
        # 系统级精确率：有风险的 pred 里，真正对上（含过宽/冲突）人工标注的比例
        "precision_rate": round(claimed_preds / len(preds), 4) if preds else 0.0,
        "high_risk_miss": high_risk_miss,
        "false_alarm": sum(1 for h in pred_hit if not h),
        "verdict_counts": dict(verdict_counts),
        "thresholds": {
            "iou_strong": config.GOLDEN_IOU_STRONG, "iou_min": iou_threshold,
            "broad_recall": config.TOO_BROAD_RECALL,
            "broad_precision": config.TOO_BROAD_PRECISION,
            "gt_map_similarity": GT_MAP_SIMILARITY,
        },
        "gt_mapped": sum(1 for x in gt_maps if x["mapped"]),
        "by_level": {k: dict(v) for k, v in by_level.items()},
        "samples": samples,
        "has_golden": bool(golden),
    }


# ==================================================================== 报告
def build_report(result: Dict[str, Any], clean_text: str = "") -> Dict[str, Any]:
    """把一次运行的原始结果收敛成一份可归档的验收报告。"""
    stats = result.get("stats") or {}
    ev = stats.get("evidence") or {}
    ok = ev.get(db.EVIDENCE_OK, 0)
    fail = ev.get(db.EVIDENCE_FAIL, 0)
    none = ev.get(db.EVIDENCE_NONE, 0)
    located_denom = ok + fail

    review = stats.get("review") or {}
    reviewed = sum(v for k, v in review.items() if k != "未复核")
    revise = review.get("修正", 0)
    reject = review.get("驳回", 0)

    report = {
        "contract_id": result.get("contract_id"),
        "file_name": result.get("file_name"),
        "terms_total": result.get("terms_total", 0),
        "risk_total": result.get("risk_total", 0),
        "needs_human": result.get("needs_human", 0),
        "duration_s": result.get("duration_s"),
        "evidence": {
            "定位成功": ok, "定位失败": fail, "缺少有效证据": none,
            "定位成功率": round(ok / located_denom, 4) if located_denom else 0.0,
        },
        "review": {
            "已复核": reviewed, "采信": review.get("采信", 0),
            "修正": revise, "驳回": reject, "复核失败": review.get("复核失败", 0),
            "修正率": round(revise / reviewed, 4) if reviewed else 0.0,
            "驳回率": round(reject / reviewed, 4) if reviewed else 0.0,
        },
        "cost": {
            "judge_tokens": stats.get("judge_total_tokens", 0),
            "tokens_per_term": round(stats.get("judge_total_tokens", 0) / max(1, result.get("terms_total", 1)), 1),
            "llm_calls": (result.get("scheduler") or {}).get("llm_calls", 0),
        },
        "recovery": {
            "branches": (result.get("scheduler") or {}).get("total", 0),
            "retried": (result.get("scheduler") or {}).get("retried", 0),
        },
    }
    report["golden"] = score_against_golden(result, clean_text)
    return report


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n===== {report['file_name']} =====")
    print(f"条款 {report['terms_total']} 条 | 风险 {report['risk_total']} 条 | "
          f"待人工 {report['needs_human']} 条 | 耗时 {report['duration_s']}s")

    ev = report["evidence"]
    print(f"\n[证据校验] 定位成功 {ev['定位成功']} / 定位失败 {ev['定位失败']} / "
          f"缺少有效证据 {ev['缺少有效证据']}  → 定位成功率 {ev['定位成功率']:.2%}")

    rv = report["review"]
    print(f"[复核动作] 已复核 {rv['已复核']}：采信 {rv['采信']} / 修正 {rv['修正']} / "
          f"驳回 {rv['驳回']} / 失败 {rv['复核失败']}  "
          f"→ 修正率 {rv['修正率']:.2%}，驳回率 {rv['驳回率']:.2%}")

    c = report["cost"]
    print(f"[成本] 判定 token {c['judge_tokens']}，单条款均 {c['tokens_per_term']}，"
          f"模型调用 {c['llm_calls']} 次")
    print(f"[调度] 分支 {report['recovery']['branches']} 个，其中重试 "
          f"{report['recovery']['retried']} 个")

    g = report["golden"]
    if g.get("has_golden"):
        vc = g.get("verdict_counts", {})
        print(f"[金标] 标注 {g['golden_total']} 条：命中 {g['golden_hit']} "
              f"(含模糊 {vc.get('ambiguous', 0)}) / 过宽待人工 {g.get('too_broad', 0)} / "
              f"冲突待裁定 {vc.get('conflict', 0)} / "
              f"漏检 {vc.get('miss', 0)} → 召回 {g['recall']:.2%}，"
              f"含过宽 {g.get('recall_with_broad', g['recall']):.2%}；"
              f"精确率 {g.get('precision_rate', 0):.2%}（误报 {g['false_alarm']}）；"
              f"高危漏检 {g['high_risk_miss']}")
        for lv, v in g["by_level"].items():
            print(f"       {lv}: 中 {v['hit']} / 过宽 {v.get('broad', 0)} / "
                  f"冲突 {v.get('conflict', 0)} / 漏 {v['miss']}")
        for s in g.get("samples", []):
            if s["verdict"] == "conflict":
                print(f"       [冲突] GT{s['gt_index']} 与 pred{s['pred_index']} 重叠 "
                      f"iou={s['iou']:.2f} R={s['recall']:.2f} ::{s['gt_text'][:26]}")
            elif s["verdict"] == "miss":
                print(f"       [漏检] GT{s['gt_index']} {s['risk_level']} "
                      f"iou={s['iou']:.2f} P={s['precision']:.2f} R={s['recall']:.2f} "
                      f"::{s['gt_text'][:26]}")
    else:
        print("[金标] 该合同无人工标注，跳过召回比对")
