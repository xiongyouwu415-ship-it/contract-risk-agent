"""db.py —— SQLite 单文件持久化底座（方案 9 数据库设计）。

六张表：
  contract_global_ctx   合同全局上下文（含归一化原文 clean_full_text）
  contract_terms        条款原文，一条一个 term_id
  judge_task_snapshot   判定任务快照：消息历史 / 状态 / 风险草稿 / 工具调用 / token
  review_task_snapshot  复核任务快照：判定结果 / 动作 / 最终结论
  evidence_result       证据校验结果：三态 + 锚点 + 相似度
  regression_pool       回归池：缺少有效证据的无效结论沉淀
  review_result         运行结果归档：一次完整审查的最终结论与验收报告（供服务层读取）

访问约定（方案 9 末段）：
  · 全局上下文对判定与复核任务只读；
  · 任务快照只由所属任务实例写入，任务之间互不干扰；
  · 所有状态变更以事务提交。

关于 clean_full_text：方案表 8 的字段建议里没有它，但 5.1 明确要求「清洗产出
clean_full_text，同时构建字符偏移索引，供证据校验对齐」。原文必须落库，
否则进程重启后无法校验锚点，因此挂在 contract_global_ctx 上（字段设计属
「详细设计阶段可细化」范围）。
"""
import json
import os
import sqlite3
import datetime
from typing import Optional, Dict, Any, List

import config

# 任务状态（方案 6.2 第三步 / 表 8 status 字段）
PENDING = "PENDING"      # 已建实例，未开始
RUNNING = "RUNNING"      # 正在推理
WAIT_LLM = "WAIT_LLM"    # 工具已回写，等待 LLM 继续思考
FINISHED = "FINISHED"    # 已产出结论
FAILED = "FAILED"        # 超过轮次上限或执行异常

# 证据校验三态（方案 7.4 表 5）
EVIDENCE_OK = "定位成功"
EVIDENCE_FAIL = "定位失败"
EVIDENCE_NONE = "缺少有效证据"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _load(text: Optional[str]) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _conn() -> sqlite3.Connection:
    """开一条通道：确保目录存在 → 连接 → 按列名取数 → WAL（边写边读不打架）。"""
    path = config.DB_PATH
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """建表，幂等。程序启动调一次即可。"""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contract_global_ctx (
                contract_id     TEXT PRIMARY KEY,
                file_name       TEXT,
                clean_text      TEXT NOT NULL,          -- 归一化干净原文，证据锚点的坐标系
                meta_json       TEXT,                   -- 甲方/乙方/标的/验收标准等元信息
                global_rules    TEXT,                   -- 全局统一规则
                custom_terms    TEXT,                   -- 自定义术语（定义区抽取）
                term_index_json TEXT,                   -- term_id -> 条款摘要
                created_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contract_terms (
                contract_id TEXT NOT NULL,
                term_id     TEXT NOT NULL,
                term_no     INTEGER NOT NULL,
                term_text   TEXT NOT NULL,
                complexity  TEXT NOT NULL,              -- simple / complex
                char_start  INTEGER NOT NULL,           -- 在 clean_full_text 中的偏移
                char_end    INTEGER NOT NULL,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (contract_id, term_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS judge_task_snapshot (
                task_id         TEXT PRIMARY KEY,
                contract_id     TEXT NOT NULL,
                term_id         TEXT NOT NULL,
                status          TEXT NOT NULL,
                max_rounds      INTEGER NOT NULL,
                cur_round       INTEGER NOT NULL DEFAULT 0,
                messages_json   TEXT,
                risk_draft_json TEXT,
                tool_calls_json TEXT,
                token_usage     INTEGER NOT NULL DEFAULT 0,
                last_error      TEXT,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_task_snapshot (
                task_id         TEXT PRIMARY KEY,
                judge_task_id   TEXT NOT NULL,
                contract_id     TEXT NOT NULL,
                term_id         TEXT NOT NULL,
                status          TEXT NOT NULL,
                max_rounds      INTEGER NOT NULL,
                cur_round       INTEGER NOT NULL DEFAULT 0,
                messages_json   TEXT,
                action          TEXT,                   -- 采信/修正/补充核验/驳回
                final_risk_json TEXT,
                token_usage     INTEGER NOT NULL DEFAULT 0,
                last_error      TEXT,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_result (
                task_id       TEXT PRIMARY KEY,         -- 与 judge_task_snapshot.task_id 一一对应
                contract_id   TEXT NOT NULL,
                term_id       TEXT NOT NULL,
                verify_status TEXT NOT NULL,            -- 定位成功 / 定位失败 / 缺少有效证据
                start_offset  INTEGER,
                end_offset    INTEGER,
                similarity    REAL NOT NULL DEFAULT 0.0,
                matched_text  TEXT,
                created_at    TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regression_pool (
                case_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                term_id     TEXT NOT NULL,
                reason      TEXT NOT NULL,
                detail      TEXT,
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_result (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                run_id      TEXT NOT NULL,
                status      TEXT NOT NULL,              -- done / failed
                result_json TEXT,                       -- 完整 final 列表
                report_json TEXT,                       -- 验收指标报告
                duration_s  REAL,
                error       TEXT,
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_terms_contract ON contract_terms(contract_id, term_no)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_result_contract ON review_result(contract_id, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_judge_contract ON judge_task_snapshot(contract_id, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_contract ON evidence_result(contract_id, verify_status)")


# ==================================================================== 全局上下文与条款
def save_global_ctx(contract_id: str, file_name: str, clean_text: str,
                    meta: Dict[str, Any], global_rules: List[str],
                    custom_terms: Dict[str, str], term_index: Dict[str, str]) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO contract_global_ctx
                 (contract_id, file_name, clean_text, meta_json, global_rules,
                  custom_terms, term_index_json, created_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(contract_id) DO UPDATE SET
                 file_name=excluded.file_name, clean_text=excluded.clean_text,
                 meta_json=excluded.meta_json, global_rules=excluded.global_rules,
                 custom_terms=excluded.custom_terms, term_index_json=excluded.term_index_json""",
            (contract_id, file_name, clean_text, _dump(meta), _dump(global_rules),
             _dump(custom_terms), _dump(term_index), _now())
        )


def load_global_ctx(contract_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM contract_global_ctx WHERE contract_id=?", (contract_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "contract_id": row["contract_id"],
        "file_name": row["file_name"],
        "clean_text": row["clean_text"],
        "meta": _load(row["meta_json"]) or {},
        "global_rules": _load(row["global_rules"]) or [],
        "custom_terms": _load(row["custom_terms"]) or {},
        "term_index": _load(row["term_index_json"]) or {},
    }


def save_terms(contract_id: str, terms: List[Dict[str, Any]]) -> None:
    """整批覆盖写条款；重跑解析时旧条款先清掉，避免残留。"""
    with _conn() as conn:
        conn.execute("DELETE FROM contract_terms WHERE contract_id=?", (contract_id,))
        conn.executemany(
            """INSERT INTO contract_terms
                 (contract_id, term_id, term_no, term_text, complexity,
                  char_start, char_end, created_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            [(contract_id, t["term_id"], t["term_no"], t["term_text"], t["complexity"],
              t["char_start"], t["char_end"], _now()) for t in terms]
        )


def load_terms(contract_id: str) -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM contract_terms WHERE contract_id=? ORDER BY term_no", (contract_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ==================================================================== 判定任务快照
def get_judge_snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM judge_task_snapshot WHERE task_id=?", (task_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["messages"] = _load(row["messages_json"]) or []
    d["risk_draft"] = _load(row["risk_draft_json"])
    d["tool_calls"] = _load(row["tool_calls_json"]) or []
    return d


def upsert_judge_snapshot(task_id: str, contract_id: str, term_id: str, status: str,
                          max_rounds: int, cur_round: int,
                          messages: List[Dict[str, Any]],
                          risk_draft: Optional[Dict[str, Any]] = None,
                          tool_calls: Optional[List[Dict[str, Any]]] = None,
                          token_usage: int = 0,
                          last_error: Optional[str] = None) -> None:
    """一个事务写完一条任务快照。任务之间互不干扰（各自 task_id 一行）。"""
    with _conn() as conn:
        conn.execute(
            """INSERT INTO judge_task_snapshot
                 (task_id, contract_id, term_id, status, max_rounds, cur_round,
                  messages_json, risk_draft_json, tool_calls_json, token_usage,
                  last_error, updated_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                 status=excluded.status, cur_round=excluded.cur_round,
                 messages_json=excluded.messages_json,
                 risk_draft_json=excluded.risk_draft_json,
                 tool_calls_json=excluded.tool_calls_json,
                 token_usage=excluded.token_usage,
                 last_error=excluded.last_error,
                 updated_at=excluded.updated_at""",
            (task_id, contract_id, term_id, status, max_rounds, cur_round,
             _dump(messages), _dump(risk_draft), _dump(tool_calls),
             token_usage, last_error, _now())
        )


def list_judge_snapshots(contract_id: str) -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM judge_task_snapshot WHERE contract_id=? ORDER BY term_id",
            (contract_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["risk_draft"] = _load(r["risk_draft_json"])
        d["messages"] = _load(r["messages_json"]) or []
        out.append(d)
    return out


# ==================================================================== 复核任务快照
def get_review_snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM review_task_snapshot WHERE task_id=?", (task_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["messages"] = _load(row["messages_json"]) or []
    d["final_risk"] = _load(row["final_risk_json"])
    return d


def upsert_review_snapshot(task_id: str, judge_task_id: str, contract_id: str, term_id: str,
                           status: str, max_rounds: int, cur_round: int,
                           messages: List[Dict[str, Any]], action: Optional[str] = None,
                           final_risk: Optional[Dict[str, Any]] = None,
                           token_usage: int = 0, last_error: Optional[str] = None) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO review_task_snapshot
                 (task_id, judge_task_id, contract_id, term_id, status, max_rounds,
                  cur_round, messages_json, action, final_risk_json, token_usage,
                  last_error, updated_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                 status=excluded.status, cur_round=excluded.cur_round,
                 messages_json=excluded.messages_json, action=excluded.action,
                 final_risk_json=excluded.final_risk_json,
                 token_usage=excluded.token_usage, last_error=excluded.last_error,
                 updated_at=excluded.updated_at""",
            (task_id, judge_task_id, contract_id, term_id, status, max_rounds,
             cur_round, _dump(messages), action, _dump(final_risk), token_usage,
             last_error, _now())
        )


def list_review_snapshots(contract_id: str) -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM review_task_snapshot WHERE contract_id=? ORDER BY term_id",
            (contract_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["final_risk"] = _load(r["final_risk_json"])
        out.append(d)
    return out


# ==================================================================== 证据校验结果
def save_evidence(task_id: str, contract_id: str, term_id: str, verify_status: str,
                  start_offset: Optional[int], end_offset: Optional[int],
                  similarity: float, matched_text: str) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO evidence_result
                 (task_id, contract_id, term_id, verify_status, start_offset,
                  end_offset, similarity, matched_text, created_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                 verify_status=excluded.verify_status,
                 start_offset=excluded.start_offset, end_offset=excluded.end_offset,
                 similarity=excluded.similarity, matched_text=excluded.matched_text""",
            (task_id, contract_id, term_id, verify_status, start_offset, end_offset,
             float(similarity), matched_text or "", _now())
        )


def load_evidence(task_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM evidence_result WHERE task_id=?", (task_id,)).fetchone()
    return dict(row) if row else None


def list_evidence(contract_id: str) -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM evidence_result WHERE contract_id=? ORDER BY term_id", (contract_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ==================================================================== 回归池
def push_regression(contract_id: str, term_id: str, reason: str,
                    detail: Optional[Dict[str, Any]] = None) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO regression_pool(contract_id, term_id, reason, detail, created_at)
               VALUES(?, ?, ?, ?, ?)""",
            (contract_id, term_id, reason, _dump(detail), _now())
        )


def list_regression(limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM regression_pool ORDER BY case_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ==================================================================== 重置
def reset_contract(contract_id: str) -> None:
    """清空一份合同的全部分析痕迹，用于 --fresh 从头重跑。

    只删这一份合同的行，不碰库里其他合同 —— 这也是「任务级写入隔离」的体现。
    """
    with _conn() as conn:
        conn.execute(
            "DELETE FROM evidence_result WHERE task_id IN "
            "(SELECT task_id FROM judge_task_snapshot WHERE contract_id=?)", (contract_id,))
        conn.execute("DELETE FROM review_task_snapshot WHERE contract_id=?", (contract_id,))
        conn.execute("DELETE FROM judge_task_snapshot WHERE contract_id=?", (contract_id,))
        conn.execute("DELETE FROM contract_terms WHERE contract_id=?", (contract_id,))
        conn.execute("DELETE FROM contract_global_ctx WHERE contract_id=?", (contract_id,))
        conn.execute("DELETE FROM regression_pool WHERE contract_id=?", (contract_id,))
        conn.execute("DELETE FROM review_result WHERE contract_id=?", (contract_id,))


# ==================================================================== 观测
def contract_stats(contract_id: str) -> Dict[str, Any]:
    """一次任务跑完后算的收尾统计，供验收指标使用（方案 12.3）。"""
    with _conn() as conn:
        judge_rows = conn.execute(
            """SELECT status, COUNT(*) AS cnt, SUM(token_usage) AS tok
               FROM judge_task_snapshot WHERE contract_id=? GROUP BY status""",
            (contract_id,)).fetchall()
        ev_rows = conn.execute(
            """SELECT verify_status, COUNT(*) AS cnt FROM evidence_result
               WHERE contract_id=? GROUP BY verify_status""", (contract_id,)).fetchall()
        act_rows = conn.execute(
            """SELECT COALESCE(action,'未复核') AS action, COUNT(*) AS cnt
               FROM review_task_snapshot WHERE contract_id=? GROUP BY action""",
            (contract_id,)).fetchall()

    stats = {
        "judge": {r["status"]: {"count": r["cnt"], "tokens": r["tok"] or 0} for r in judge_rows},
        "evidence": {r["verify_status"]: r["cnt"] for r in ev_rows},
        "review": {r["action"]: r["cnt"] for r in act_rows},
    }
    stats["judge_total_tokens"] = sum(v["tokens"] for v in stats["judge"].values())
    return stats


# ==================================================================== 结果归档
def save_result(contract_id: str, run_id: str, status: str,
                result: Optional[Dict[str, Any]] = None,
                report: Optional[Dict[str, Any]] = None,
                duration_s: float = 0.0, error: Optional[str] = None) -> None:
    """归档一次运行的最终结论。

    为什么不只靠快照还原：快照记录的是「每个任务的推理过程」，而最终结论要经过
    证据分流、复核动作、失败兜底三处收敛，重新拼装容易与 pipeline 口径漂移。
    直接把落定结果存一份，读的时候就是它，不二次推导。
    """
    with _conn() as conn:
        conn.execute(
            """INSERT INTO review_result
                 (contract_id, run_id, status, result_json, report_json,
                  duration_s, error, created_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            (contract_id, run_id, status, _dump(result or {}),
             _dump(report or {}), float(duration_s or 0.0), error, _now())
        )


def load_latest_result(contract_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM review_result WHERE contract_id=? ORDER BY id DESC LIMIT 1",
            (contract_id,)).fetchone()
    if row is None:
        return None
    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "result": _load(row["result_json"]) or {},
        "report": _load(row["report_json"]) or {},
        "duration_s": row["duration_s"],
        "error": row["error"],
        "created_at": row["created_at"],
    }


def list_latest_results() -> Dict[str, Dict[str, Any]]:
    """每份合同只取最近一条，供列表页一次性展示，避免 N+1 查询。"""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT r.contract_id, r.status, r.created_at, r.result_json
               FROM review_result r
               JOIN (SELECT contract_id, MAX(id) AS mid FROM review_result GROUP BY contract_id) m
                 ON r.id = m.mid""").fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        data = _load(r["result_json"]) or {}
        out[r["contract_id"]] = {
            "status": r["status"],
            "created_at": r["created_at"],
            "terms_total": data.get("terms_total", 0),
            "risk_total": data.get("risk_total", 0),
            "needs_human": data.get("needs_human", 0),
        }
    return out


def list_contracts() -> List[Dict[str, Any]]:
    """库里出现过的合同：条款数、判定完成数、证据条数。用于列表页概览。"""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT c.contract_id, c.file_name, COUNT(t.term_id) AS terms,
                      MAX(c.created_at) AS created_at
               FROM contract_global_ctx c
               LEFT JOIN contract_terms t ON t.contract_id = c.contract_id
               GROUP BY c.contract_id""").fetchall()
    return [dict(r) for r in rows]
