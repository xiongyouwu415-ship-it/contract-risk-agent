"""run_demo.py —— 命令行入口：对样例合同跑完整流水线并输出验收报告。

用法：
  python run_demo.py                          # 跑 rule/ 下第一份合同
  python run_demo.py --limit 3                # 跑 3 份
  python run_demo.py --contract 合同04_劳动合同_陷阱测试稿.txt
  python run_demo.py --parse-only             # 只跑解析（不需要密钥）
  python run_demo.py --fresh                  # 丢弃已有快照从头跑

前置：环境变量 DEEPSEEK_API_KEY（判定与复核要调模型）。
"""
import argparse
import asyncio
import json
import logging
import os
from typing import List

import config
import db
import metrics
import parsing
import pipeline
from llm import LLMClient, TokenBucket

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

REPORT_DIR = os.path.join(config.BASE_DIR, "reports")


def list_contracts(limit: int = 0) -> List[str]:
    """列出 rule/ 下的样例合同（排序固定，保证可复现）。"""
    names = sorted(n for n in os.listdir(config.SAMPLE_CONTRACT_DIR) if n.endswith(".txt"))
    return names[:limit] if limit else names


def read_contract(name: str) -> str:
    with open(os.path.join(config.SAMPLE_CONTRACT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


async def run(args) -> None:
    if args.db:
        config.DB_PATH = args.db
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    db.init_db()

    targets = ([args.contract] if args.contract else list_contracts(args.limit))
    if not targets:
        print("rule/ 下没有找到 .txt 样例合同")
        return

    llm = None
    if not args.parse_only:
        llm = LLMClient(config.api_key(), limiter=TokenBucket())

    summaries = []
    for name in targets:
        contract_id = os.path.splitext(name)[0]
        if args.fresh:
            db.reset_contract(contract_id)

        raw = read_contract(name)

        # 只跑解析：不判定、不复核、不写报告，专门用来检查切分质量
        if args.parse_only:
            parsed = await parsing.parse_contract(
                contract_id, name, raw, llm=None, use_llm=False, persist=True)
            print(f"\n===== {name} 解析结果 =====")
            print(f"清洗后 {len(parsed['clean_text'])} 字，条款 {len(parsed['terms'])} 条")
            for t in parsed["terms"]:
                print(f"  {t['term_id']} [{t['complexity']:<7}] "
                      f"{t['char_start']:>5}-{t['char_end']:<5} {t['term_text'][:42]!r}")
            print(f"合同要素：{parsed['global_ctx']['meta']}")
            if parsed["global_ctx"]["custom_terms"]:
                print(f"自定义术语：{parsed['global_ctx']['custom_terms']}")
            continue

        result = await pipeline.review_contract(
            contract_id, name, raw, llm=llm,
            persist=True, resume=not args.fresh,
            use_llm_parse=not args.no_llm_parse)

        report = metrics.build_report(result, _clean_text(contract_id))
        metrics.print_report(report)
        summaries.append(report)

        os.makedirs(REPORT_DIR, exist_ok=True)
        out = os.path.join(REPORT_DIR, f"{contract_id}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"report": report, "final": result["final"]},
                      f, ensure_ascii=False, indent=2)
        print(f"详细结论已写入：{out}")

    if len(summaries) > 1:
        _print_overview(summaries)


def _clean_text(contract_id: str) -> str:
    ctx = db.load_global_ctx(contract_id)
    return ctx["clean_text"] if ctx else ""


def _print_overview(reports: List[dict]) -> None:
    print("\n===== 批次总览 =====")
    print(f"{'合同':<28}{'条款':>6}{'风险':>6}{'定位成功率':>12}{'召回':>10}{'耗时':>8}")
    for r in reports:
        print(f"{r['file_name'][:26]:<28}{r['terms_total']:>6}{r['risk_total']:>6}"
              f"{r['evidence']['定位成功率']:>12.2%}"
              f"{r['golden']['recall']:>10.2%}{r['duration_s']:>8.1f}s")
    total = sum(r['terms_total'] for r in reports)
    tok = sum(r['cost']['judge_tokens'] for r in reports)
    print(f"合计条款 {total} 条，判定 token {tok}，单条款均 "
          f"{tok / max(1, total):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="合同风险审查多 Agent 系统 —— 演示入口")
    parser.add_argument("--contract", help="rule/ 下的合同文件名")
    parser.add_argument("--limit", type=int, default=1, help="处理前 N 份，0 表示全部")
    parser.add_argument("--parse-only", action="store_true", help="只跑解析，不调模型")
    parser.add_argument("--no-llm-parse", action="store_true",
                        help="解析阶段用启发式判定，省调用（判定/复核仍然走模型）")
    parser.add_argument("--fresh", action="store_true", help="丢弃已有快照从头跑")
    parser.add_argument("--db", help="覆盖数据库路径")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
