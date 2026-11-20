"""tests/selftest.py —— 离线自测：不联网、不花钱，验证确定性模块与调度语义。

覆盖范围（对应方案 12.1 单元测试）：
  1 归一化与偏移映射         evidence.normalize
  2 滑动窗口模糊匹配与三路分流 evidence.find_span / verify
  3 解析：清洗、锚点粗切、组装、复杂度打标
  4 模型输出 JSON 解析的健壮性 judging.parse_json_object
  5 工具注册表与分发器（含未注册工具拒绝）
  6 SQLite 快照读写与 reset
  7 调度器：Fan-out 隔离、只重跑失败分支、幂等跳过
  8 端到端：假模型跑通 解析→判定→校验→复核，并验证二次运行零调用

运行：python tests/selftest.py
"""
import asyncio
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# 先把数据库指到临时文件，避免污染正式库
_TMP_DIR = tempfile.mkdtemp(prefix="contract_agent_selftest_")
config.DB_PATH = os.path.join(_TMP_DIR, "selftest.db")

import db                                  # noqa: E402
import evidence as ev                      # noqa: E402
import judging                             # noqa: E402
import metrics                             # noqa: E402
import parsing                             # noqa: E402
import pipeline                            # noqa: E402
import tools                               # noqa: E402
from llm import ChatResult                 # noqa: E402
from scheduler import BranchResult, Scheduler   # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  [ok] {name}")
    else:
        FAIL.append(f"{name} {detail}")
        print(f"  [FAIL] {name} {detail}")


# ==================================================================== 1 归一化
def test_normalize() -> None:
    print("\n[1] 归一化与偏移映射")
    origin = "甲　方：　张 三\r\n\r\n乙方：李四"
    norm, index_map = ev.normalize(origin)
    check("连续空白压成一个", "  " not in norm, repr(norm))
    check("全角空格转半角", "　" not in norm, repr(norm))
    check("映射表长度一致", len(norm) == len(index_map))

    # 映射必须能还原：归一化后的第 i 个字符，应等于原文 index_map[i] 处字符归一化的结果
    import unicodedata
    recon_ok = True
    for i, orig_i in enumerate(index_map):
        src = unicodedata.normalize("NFKC", origin[orig_i]) or " "
        if norm[i] == " ":
            recon_ok &= src.strip() == ""
        else:
            recon_ok &= src[0] == norm[i]
    check("归一化字符可由映射还原", recon_ok)
    check("映射单调递增", all(a < b for a, b in zip(index_map, index_map[1:])))


# ==================================================================== 2 模糊匹配
def test_evidence() -> None:
    print("\n[2] 滑动窗口模糊匹配与三路分流")
    text = ("一、标的物与质量标准\n　　甲方向乙方采购钣金结构件，规格型号详见附件《采购清单》。\n"
            "二、违约责任\n　　乙方逾期交货的，每逾期一日按合同总金额的0.5%支付违约金。\n"
            "三、争议解决\n　　因本合同发生的争议，由乙方所在地人民法院管辖。")

    exact = "乙方逾期交货的，每逾期一日按合同总金额的0.5%支付违约金。"
    r = ev.verify(text, exact)
    check("精确证据 → 定位成功", r["verify_status"] == db.EVIDENCE_OK, r)
    check("锚点能切回原文", text[r["start"]:r["end"]] == exact,
          repr(text[r["start"]:r["end"]]))

    fuzzy = "乙方逾期交货的 每逾期1日按合同总金额0.5%支付违约金"
    r2 = ev.verify(text, fuzzy)
    check("模糊证据仍能定位", r2["verify_status"] == db.EVIDENCE_OK, r2)
    check("模糊证据相似度 < 1", r2["similarity"] < 1.0, r2["similarity"])

    r3 = ev.verify(text, "甲方应在合同签订后十个工作日内支付预付款")
    check("无关证据 → 缺少有效证据", r3["verify_status"] == db.EVIDENCE_NONE, r3)

    r4 = ev.verify(text, "")
    check("空证据 → 缺少有效证据", r4["verify_status"] == db.EVIDENCE_NONE, r4)


# ==================================================================== 3 解析
def test_parsing() -> None:
    print("\n[3] 解析：清洗 / 锚点粗切 / 组装 / 复杂度")
    raw = "合同标题\n-----\n一、标的物\n　　甲方向乙方采购结构件，数量详见附件。\n二、违约责任\n　　乙方逾期交货按日万分之五支付违约金。\n第 1 页"
    clean = parsing.clean_text(raw)
    check("剔除页眉分割线", "-----" not in clean)
    check("剔除页码行", "第 1 页" not in clean)
    check("正文保留", "违约责任" in clean)

    cands = parsing.split_candidates(clean)
    check("锚点切出多个候选", len(cands) >= 2, f"got {len(cands)}")
    check("候选偏移落在原文内", all(clean[s:e] == piece for s, e, piece in cands))

    labels = ["ignore"] + ["valid"] * (len(cands) - 1)
    terms = parsing.assemble(cands, labels)
    check("组装后条款非空", len(terms) >= 1, f"got {len(terms)}")

    merged = parsing.assemble(cands, ["ignore", "valid", "merge"] + ["ignore"] * (len(cands) - 3))
    if len(merged) >= 1:
        check("merge 并到上一条", "\n" in merged[0]["text"] or len(merged) == 1)

    check("短条款判 simple", parsing.tag_complexity("乙方应在三日内交货。") == "simple")
    check("带金额比例判 complex",
          parsing.tag_complexity("乙方逾期交货的，每逾期一日按合同总金额的0.5%支付违约金，"
                                 "并赔偿甲方因此遭受的全部损失，具体计算方式详见附件三。") == "complex")


# ==================================================================== 4 JSON 解析
def test_json_parse() -> None:
    print("\n[4] 模型输出 JSON 解析")
    cases = [
        ('{"action":"final","has_risk":true}', True),
        ('```json\n{"action":"final"}\n```', True),
        ('先说明一下：\n{"action":"tool_call","tool":"search_law","args":{"query":"x"}}\n以上。', True),
        ('{"a":1} {"b":2}', True),
        ('完全不是 JSON', False),
        ('', False),
    ]
    for text, ok in cases:
        got = judging.parse_json_object(text)
        check(f"解析 {text[:24]!r}", (got is not None) == ok, repr(got))
    check("取第一个对象", judging.parse_json_object('{"a":1}{"b":2}') == {"a": 1})

    # Prompt 模板必须能 format 成功，且花括号转义没漏（漏了真跑时才会炸）
    import prompts
    import tools as tools_mod
    js = prompts.JUDGE_SYSTEM.format(tool_list=tools_mod.TOOL_LIST_FOR_PROMPT)
    check("判定 System Prompt 可格式化", "{" in js and "{tool_list}" not in js)
    rs = prompts.REVIEW_SYSTEM.format(tool_list=tools_mod.TOOL_LIST_FOR_PROMPT)
    check("复核 System Prompt 可格式化", "accept" in rs and "{tool_list}" not in rs)
    ju = prompts.JUDGE_USER_TEMPLATE.format(global_ctx="g", term_id="t", complexity="simple",
                                            term_text="x", extra="")
    check("判定 User Prompt 可格式化", "t" in ju and "{term_text}" not in ju)
    ru = prompts.REVIEW_USER_TEMPLATE.format(global_ctx="g", term_id="t", term_text="x",
                                             judge_result="j", verify_status="v",
                                             anchor_text="a", similarity="0.9", extra="")
    check("复核 User Prompt 可格式化", "0.9" in ru and "{similarity}" not in ru)
    pb = prompts.PARSE_BOUNDARY.format(prev_text="p", current_text="c")
    check("解析 Prompt 只要求标签", "valid" in pb and "ignore" in pb)


# ==================================================================== 5 工具层
def test_tools() -> None:
    print("\n[5] 工具注册表与分发器")
    r = tools.dispatch("calc_amount", {"base_amount": 100000, "rate": 1.5, "rate_type": "total"})
    check("金额核算可用", r.get("ok") and r.get("amount") == 150000.0, r)
    check("超 130% 给出提示", any("130%" in n for n in r.get("notes", [])), r.get("notes"))

    r2 = tools.dispatch("search_law", {"query": "违约金过分高于损失"})
    check("法条检索可用", r2.get("ok") and len(r2.get("hits", [])) > 0, r2.get("hits", [])[:1])

    r3 = tools.dispatch("check_redline", {"text": "甲方不可撤销地确认，对乙方单方出具的结算单不持异议。"})
    check("红线查询命中", r3.get("ok") and len(r3.get("hits", [])) > 0, r3.get("hits"))

    try:
        tools.dispatch("drop_database", {})
        check("未注册工具被拒绝", False)
    except tools.ToolError as e:
        check("未注册工具被拒绝", True, str(e))

    try:
        tools.dispatch("search_law", {})
        check("缺必填参数被拒绝", False)
    except tools.ToolError:
        check("缺必填参数被拒绝", True)


# ==================================================================== 6 数据库
def test_db() -> None:
    print("\n[6] SQLite 快照读写")
    db.init_db()
    cid = "TEST-CONTRACT"
    db.save_global_ctx(cid, "t.txt", "原文内容", {"甲方": "A"}, ["规则1"], {"标的": "货物"},
                       {"TEST-CONTRACT-T001": "摘要"})
    db.save_terms(cid, [{"term_id": "TEST-CONTRACT-T001", "term_no": 1, "term_text": "条款一",
                         "complexity": "simple", "char_start": 0, "char_end": 3}])
    ctx = db.load_global_ctx(cid)
    terms = db.load_terms(cid)
    check("全局上下文落库", ctx and ctx["clean_text"] == "原文内容")
    check("条款落库", len(terms) == 1 and terms[0]["term_id"].endswith("T001"))

    db.upsert_judge_snapshot("T-1", cid, "TEST-CONTRACT-T001", db.FINISHED, 4, 2,
                             [{"role": "user", "content": "hi"}],
                             risk_draft={"has_risk": True}, token_usage=100)
    snap = db.get_judge_snapshot("T-1")
    check("任务快照落库", snap and snap["status"] == db.FINISHED and snap["token_usage"] == 100)
    check("消息历史可反序列化", snap["messages"][0]["content"] == "hi")

    db.save_evidence("T-1", cid, "TEST-CONTRACT-T001", db.EVIDENCE_OK, 0, 3, 0.95, "条款一")
    check("证据结果落库", db.load_evidence("T-1")["verify_status"] == db.EVIDENCE_OK)

    db.push_regression(cid, "TEST-CONTRACT-T001", "缺证")
    check("回归池落库", len(db.list_regression()) >= 1)

    db.reset_contract(cid)
    check("reset 后条款清空", db.load_terms(cid) == [])
    check("reset 后快照清空", db.get_judge_snapshot("T-1") is None)


# ==================================================================== 7 调度器
def test_scheduler() -> None:
    print("\n[7] 调度器：并行隔离 / 只重跑失败 / 幂等跳过")
    sched = Scheduler(max_concurrent=2, max_retry=2, backoff=0.01)
    calls = {"n": 0}

    async def worker(item):
        calls["n"] += 1
        if item == "bad" and calls["n"] <= 2:
            raise RuntimeError("boom")
        return f"ok:{item}"

    items = ["good", "bad", "good2"]
    results = asyncio.run(sched.fan_out(items, worker, name_of=str))
    check("单分支异常不影响其他分支",
          sum(1 for r in results if r.ok) == 2, [r.ok for r in results])

    async def _retry():
        return await sched.retry_failed(results, items, worker, name_of=str)

    results = asyncio.run(_retry())
    check("失败分支被重试成功", all(r.ok for r in results), [(r.key, r.ok) for r in results])
    check("重试次数被记录", any(r.attempts > 1 for r in results))

    # 幂等：is_done 为真时不再重跑
    async def always_fail(item):
        raise RuntimeError("still boom")

    res2 = asyncio.run(sched.fan_out(["x"], always_fail, name_of=str))
    before = calls["n"]

    async def _retry2():
        return await sched.retry_failed(res2, ["x"], always_fail, name_of=str,
                                        is_done=lambda item: True)

    res2 = asyncio.run(_retry2())
    check("快照已完成则跳过重跑", res2[0].ok and calls["n"] == before, res2[0])


# ==================================================================== 8 端到端（假模型）
class FakeLLM:
    """按 Prompt 特征给出确定性回复，用来在无网络环境跑通整条链路。

    行为模拟真实的模型：判定先调一次红线工具，拿到结果后再给 final，
    且 evidence 逐字取自条款原文 —— 这样证据校验必然定位成功。
    """

    def __init__(self):
        self.calls = 0

    async def chat(self, messages):
        self.calls += 1
        convo = "\n".join(str(m.get("content", "")) for m in messages)

        if "你是合同条款语义分析器" in convo:
            return ChatResult("valid", 10, 2)

        if "你是合同风险复核专员" in convo:
            return ChatResult('{"action":"accept"}', 20, 3)

        if "你是合同风险审查专家" in convo:
            if "返回结果" in convo:                      # 已经拿到工具结果 → 给结论
                m = re.search(r"当前条款（term_id=([^）]+)，复杂度=[^）]*）：\n(.*?)\n\n请开始判断",
                              convo, re.S)
                term_text = (m.group(2).strip() if m else "").split("\n")[0]
                evidence = term_text[:60].strip()
                payload = {"action": "final", "has_risk": True, "risk_level": "高",
                           "reason": "该条款将风险转移给我方，显失公平",
                           "basis": "民法典第497条", "evidence": evidence}
                return ChatResult(json.dumps(payload, ensure_ascii=False), 30, 5)
            return ChatResult('{"action":"tool_call","tool":"check_redline",'
                              '"args":{"text":"质量瑕疵 免责"}}', 25, 4)

        return ChatResult("{}", 1, 1)


def test_pipeline_e2e() -> None:
    print("\n[8] 端到端：解析 → 判定 → 证据校验 → 复核")
    db.init_db()
    name = "合同01_货物采购合同_陷阱测试稿.txt"
    path = os.path.join(config.SAMPLE_CONTRACT_DIR, name)
    if not os.path.exists(path):
        print("  [skip] 样例合同不存在，跳过端到端")
        return
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    fake = FakeLLM()
    result = asyncio.run(pipeline.review_contract(
        "SELFTEST-01", name, raw, llm=fake, persist=True, resume=False,
        use_llm_parse=True))

    check("解析出条款", result["terms_total"] > 0, result["terms_total"])
    check("判定出风险", result["risk_total"] > 0, result["risk_total"])
    check("工具被调用过", any(f.get("action") != "无风险" for f in result["final"]))

    ev_stats = result["stats"].get("evidence", {})
    check("证据校验有定位成功记录", ev_stats.get(db.EVIDENCE_OK, 0) > 0, ev_stats)
    check("复核动作已记录", "采信" in (result["stats"].get("review") or {}),
          result["stats"].get("review"))

    report = metrics.build_report(result, db.load_global_ctx("SELFTEST-01")["clean_text"])
    check("报告含定位成功率", "定位成功率" in report["evidence"], report["evidence"])
    check("金标比对可执行", report["golden"]["has_golden"] is True, report["golden"])

    # 二次运行：全部命中已完成快照，模型调用次数不应再增长
    before = fake.calls
    result2 = asyncio.run(pipeline.review_contract(
        "SELFTEST-01", name, raw, llm=fake, persist=True, resume=True,
        use_llm_parse=True))
    check("二次运行零模型调用（快照复用）", fake.calls == before,
          f"{before} → {fake.calls}")
    check("二次运行结论一致", result2["risk_total"] == result["risk_total"])


# ==================================================================== 9 金标评测
def test_golden_metrics() -> None:
    print("\n[9] 金标评测：IoU / Precision / Recall 与范围过宽识别")
    om, cls = metrics.overlap_metrics, metrics.classify_pair

    # —— 三指标计算 ——
    m = om((100, 500), (200, 290))          # pred 400 字包含 GT 90 字
    check("GT完整含于pred：Recall=1", m["recall"] == 1.0, m)
    check("GT完整含于pred：Precision=90/400", m["precision"] == 0.225, m)
    check("GT完整含于pred：IoU=90/400", m["iou"] == 0.225, m)

    m = om((200, 290), (100, 500))          # pred 90 字完整含于 GT 400 字
    check("pred完整含于GT：Precision=1", m["precision"] == 1.0, m)
    check("pred完整含于GT：Recall=90/400", m["recall"] == 0.225, m)

    m = om((0, 50), (100, 150))             # 完全不重合
    check("不重合：三指标全0", m["iou"] == 0 and m["precision"] == 0
          and m["recall"] == 0 and m["inter"] == 0, m)

    m = om((0, 100), (50, 150))             # 部分重合：inter 50, union 150
    check("部分重合：IoU=1/3", abs(m["iou"] - 1 / 3) < 1e-3, m)
    check("部分重合：P=R=0.5", m["precision"] == 0.5 and m["recall"] == 0.5, m)

    for bad in (None, (), (0, 0), (10, 5), (-5, 10), ("a", "b")):
        try:
            m1, m2 = om(bad, (0, 10)), om((0, 10), bad)
        except Exception as e:              # noqa: BLE001
            check(f"异常区间 {bad!r} 不抛错", False, repr(e))
            break
    else:
        check("空/倒置/越界区间返回全0不抛错",
              m1["iou"] == 0 and m2["recall"] == 0, (m1, m2))

    # —— 档位判定 ——
    check("IoU≥0.8 → match", cls(om((0, 100), (10, 90))) == "match", None)   # 0.888
    check("0.3≤IoU<0.8 → ambiguous", cls(om((0, 100), (50, 150))) == "ambiguous", None)
    m = om((100, 500), (200, 290))          # iou 0.225, recall 1.0, precision 0.225
    check("GT全被覆盖但引用过宽 → too_broad", cls(m) == "too_broad", m)
    m = om((80, 200), (0, 100))             # inter 20, iou 0.1, recall 0.2
    check("低重叠低召回 → miss", cls(m) == "miss", m)
    check("阈值可配置：调低broad_recall翻转档位",
          cls(m, broad_recall=0.15) == "too_broad", m)
    check("阈值可配置：调高iou_strong翻转档位",
          cls(om((0, 100), (10, 90)), iou_strong=0.95) == "ambiguous", None)

    # —— score_against_golden：沿用原贪心一对一匹配 ——
    syn_gt = [
        {"start_idx": 100, "end_idx": 190, "risk_level": "高", "text": "甲" * 90},
        {"start_idx": 400, "end_idx": 435, "risk_level": "中", "text": "乙" * 35},
    ]
    syn_final = [
        {"has_risk": True, "evidence": "E1", "anchor_start": 100, "anchor_end": 500},
        {"has_risk": True, "evidence": "E2", "anchor_start": 400, "anchor_end": 470},
        {"has_risk": True, "evidence": "E3", "anchor_start": 1000, "anchor_end": 1100},
        {"has_risk": False, "evidence": "E4", "anchor_start": 0, "anchor_end": 10},
    ]
    orig_loader = metrics.load_golden
    metrics.load_golden = lambda fn: syn_gt
    try:
        res = metrics.score_against_golden(
            {"file_name": "synthetic", "final": syn_final}, "")
    finally:
        metrics.load_golden = orig_loader

    check("贪心一对一：GT0判过宽、GT1判模糊", res["golden_hit"] == 1
          and res["too_broad"] == 1, res)
    check("系统级精确率=认领pred数/风险pred数",
          abs(res["precision_rate"] - 2 / 3) < 1e-3, res["precision_rate"])
    check("过宽不计入严格召回", res["recall"] == 0.5, res)
    check("含过宽的总召回=1", res["recall_with_broad"] == 1.0, res)
    check("过宽不计为误报", res["false_alarm"] == 1, res)
    check("每条样本带 IoU/P/R/档位", all(
        {"iou", "precision", "recall", "verdict"} <= set(s)
        for s in res["samples"]), res["samples"])
    s0 = res["samples"][0]
    check("过宽样本明细数值正确", s0["verdict"] == "too_broad"
          and s0["iou"] == 0.225 and s0["recall"] == 1.0, s0)
    check("高危漏检为0（本例无漏检）", res["high_risk_miss"] == 0, res)

    # —— 漏检样本明细：无 pred 覆盖的 GT 也要给出对照指标 ——
    syn_gt2 = [{"start_idx": 0, "end_idx": 50, "risk_level": "高", "text": "丙" * 50}]
    metrics.load_golden = lambda fn: syn_gt2
    try:
        res2 = metrics.score_against_golden(
            {"file_name": "synthetic", "final": [
                {"has_risk": True, "anchor_start": 1000, "anchor_end": 1060}]}, "")
    finally:
        metrics.load_golden = orig_loader
    check("无覆盖GT计漏检", res2["golden_hit"] == 0 and res2["recall"] == 0.0, res2)
    check("漏检样本有明细与高危统计", res2["samples"]
          and res2["samples"][0]["verdict"] == "miss"
          and res2["high_risk_miss"] == 1, res2["samples"])

    # —— 坐标系映射：金标在 raw 坐标，pred 锚点在 clean 坐标 ——
    raw = ("==== 第 1 页 ====\n\n"                      # 页眉行，清洗后删除
           "第一条  质量标准：符合国家标准。\n\n\n\n"      # 连续空行被压缩
           "第二条  保密条款：乙方不得泄露甲方商业秘密。\n"
           "第三条  违约责任：违约方赔偿损失。   \n")     # 行尾空格被去掉
    clean = parsing.clean_text(raw)
    check("构造的测试文本清洗后确实变短", len(clean) < len(raw), (len(raw), len(clean)))
    # 金标按【raw 坐标】标注“乙方不得泄露甲方商业秘密”
    gt_pos = raw.find("乙方不得泄露甲方商业秘密")
    syn_gt3 = [{"start_idx": gt_pos, "end_idx": gt_pos + 13,
                "risk_level": "高", "text": "乙方不得泄露甲方商业秘密"}]
    # pred 锚点按【clean 坐标】定位同一句话，但引用范围过宽（覆盖到第三条）
    c_pos = clean.find("第二条")
    syn_final3 = [{"has_risk": True, "evidence": "保密条款",
                   "anchor_start": c_pos, "anchor_end": len(clean)}]
    metrics.load_golden = lambda fn: syn_gt3
    try:
        res3 = metrics.score_against_golden(
            {"file_name": "synthetic", "final": syn_final3}, clean)
    finally:
        metrics.load_golden = orig_loader
    s3 = res3["samples"][0]
    check("金标成功映射到clean坐标", s3["gt_mapped"] is True, s3)
    check("映射后坐标与clean偏移一致", s3["gt_span"][0] == clean.find("乙方不得泄露"), s3)
    check("跨坐标系比对能命中", res3["golden_hit"] + res3["too_broad"] == 1, res3)
    check("过宽判定：GT被完整覆盖但引用过宽", s3["verdict"] == "too_broad"
          and s3["recall"] == 1.0, s3)

    # 映射失败时退回 raw 坐标并打标记
    syn_gt4 = [{"start_idx": 0, "end_idx": 10, "risk_level": "中",
                "text": "这句话在clean文本里根本不存在xyz"}]
    metrics.load_golden = lambda fn: syn_gt4
    try:
        res4 = metrics.score_against_golden(
            {"file_name": "synthetic", "final": []}, clean)
    finally:
        metrics.load_golden = orig_loader
    check("映射失败退回raw坐标并标记", res4["samples"][0]["gt_mapped"] is False
          and res4["gt_mapped"] == 0, res4["samples"])

    # —— 匹配冲突：一条 pred 同时盖住两条 GT，第二条不算漏检 ——
    syn_gt5 = [
        {"start_idx": 100, "end_idx": 190, "risk_level": "高", "text": "甲" * 90},
        {"start_idx": 140, "end_idx": 230, "risk_level": "高", "text": "乙" * 90},
    ]
    # 一条 130 字的 pred 覆盖两条 GT（对每条的 IoU 均为 90/130≈0.69 模糊档）
    syn_final5 = [{"has_risk": True, "evidence": "E",
                   "anchor_start": 100, "anchor_end": 230}]
    metrics.load_golden = lambda fn: syn_gt5
    try:
        res5 = metrics.score_against_golden(
            {"file_name": "synthetic", "final": syn_final5}, "")
    finally:
        metrics.load_golden = orig_loader
    check("一条证据盖两条GT：第一条命中", res5["golden_hit"] == 1, res5)
    check("冲突场景精确率=1（唯一pred被认领）", res5["precision_rate"] == 1.0, res5)
    check("第二条记冲突而非漏检", res5["verdict_counts"].get("conflict") == 1
          and res5["verdict_counts"].get("miss", 0) == 0, res5)
    s5 = res5["samples"][1]
    check("冲突样本指向占用它的pred且指标正确",
          s5["verdict"] == "conflict" and s5["pred_index"] == 0
          and abs(s5["iou"] - 90 / 130) < 1e-3 and s5["recall"] == 1.0, s5)
    check("冲突不计入高危漏检", res5["high_risk_miss"] == 0, res5)
    check("冲突不影响误报统计", res5["false_alarm"] == 0, res5)

    # 没有任何 pred 覆盖时仍然是 miss（冲突逻辑不能吞掉真漏检）
    syn_gt6 = [{"start_idx": 5000, "end_idx": 5090, "risk_level": "高",
                "text": "丙" * 90}]
    metrics.load_golden = lambda fn: syn_gt5 + syn_gt6
    try:
        res6 = metrics.score_against_golden(
            {"file_name": "synthetic", "final": syn_final5}, "")
    finally:
        metrics.load_golden = orig_loader
    check("无覆盖者仍计漏检", res6["verdict_counts"].get("miss") == 1
          and res6["verdict_counts"].get("conflict") == 1, res6)
    check("真漏检仍计入高危漏检", res6["high_risk_miss"] == 1, res6)


# ==================================================================== main
def main() -> int:
    print("=" * 64)
    print(f"离线自测（数据库：{config.DB_PATH}）")
    print("=" * 64)
    test_normalize()
    test_evidence()
    test_parsing()
    test_json_parse()
    test_tools()
    test_db()
    test_scheduler()
    test_golden_metrics()
    test_pipeline_e2e()

    print("\n" + "=" * 64)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        for f in FAIL:
            print(f"  失败：{f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
