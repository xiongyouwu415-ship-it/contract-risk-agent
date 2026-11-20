"""inspect_db.py —— 命令行查看快照与回归池，替代原来的 _check_db.py。

方案 9 的六张表都能在这里看清，排查「哪条条款卡在哪一轮、为什么进回归池」时用。

用法：
  python inspect_db.py                      # 最近一份合同的概况
  python inspect_db.py --contract <id>      # 指定合同
  python inspect_db.py --contract <id> --detail   # 逐条列出条款与结论
  python inspect_db.py --regression         # 只看回归池
"""
import argparse
import json

import config
import db


def show_contract(contract_id: str, detail: bool = False) -> None:
    ctx = db.load_global_ctx(contract_id)
    if ctx is None:
        print(f"库里没有合同 {contract_id}")
        return

    terms = db.load_terms(contract_id)
    judges = {s["term_id"]: s for s in db.list_judge_snapshots(contract_id)}
    reviews = {s["term_id"]: s for s in db.list_review_snapshots(contract_id)}
    evidences = {e["term_id"]: e for e in db.list_evidence(contract_id)}

    print(f"\n【合同】{contract_id}")
    print(f"  文件：{ctx['file_name']}  归一化原文 {len(ctx['clean_text'])} 字")
    print(f"  要素：{json.dumps(ctx['meta'], ensure_ascii=False)}")
    print(f"  条款：{len(terms)} 条")

    stats = db.contract_stats(contract_id)
    print(f"  判定：{stats.get('judge')}")
    print(f"  证据：{stats.get('evidence')}")
    print(f"  复核：{stats.get('review')}")

    print(f"\n  {'条款':<8}{'复杂度':<9}{'判定':<10}{'轮次':<6}{'证据':<14}{'复核':<8}")
    print("  " + "-" * 82)
    for t in terms:
        tid = t["term_id"]
        j = judges.get(tid) or {}
        e = evidences.get(tid) or {}
        r = reviews.get(tid) or {}
        short = "T" + tid.split("-T")[-1] if "-T" in tid else tid
        print(f"  {short:<8}{t['complexity']:<9}{(j.get('status') or '-'):<10}"
              f"{str(j.get('cur_round', '-')):<5}{(e.get('verify_status') or '-'):<14}"
              f"{(r.get('action') or '-'):<8}")
        if detail:
            print(f"      条款：{t['term_text'][:70]!r}")
            if j.get("last_error"):
                print(f"      判定错误：{j['last_error'][:80]}")
            risk = (r.get("final_risk") or j.get("risk_draft") or {})
            if risk:
                print(f"      结论：{risk.get('risk_level', '')} · {risk.get('reason', '')[:60]}")


def show_regression() -> None:
    rows = db.list_regression(200)
    if not rows:
        print("回归池为空")
        return
    print(f"\n【回归池】共 {len(rows)} 条")
    for r in rows:
        print(f"  #{r['case_id']:<4} {r['contract_id']} / {r['term_id']}  {r['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="查看快照与回归池")
    parser.add_argument("--contract", help="合同 id（默认取库里最近一份）")
    parser.add_argument("--detail", action="store_true", help="逐条展开条款与结论")
    parser.add_argument("--regression", action="store_true", help="只看回归池")
    parser.add_argument("--db", help="覆盖数据库路径")
    args = parser.parse_args()

    if args.db:
        config.DB_PATH = args.db
    db.init_db()

    if args.regression:
        show_regression()
        return

    contract_id = args.contract
    if not contract_id:
        import sqlite3
        with db._conn() as conn:
            row = conn.execute(
                "SELECT contract_id FROM contract_global_ctx ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            print("库里还没有任何合同，先跑 run_demo.py")
            return
        contract_id = row["contract_id"]

    show_contract(contract_id, detail=args.detail)


if __name__ == "__main__":
    main()
