"""server.py —— FastAPI 服务层：把多 Agent 流水线包成 HTTP 接口，并托管前端。

启动：
  python server.py                    # 有密钥走真模型，没密钥自动降级为离线模拟
  python server.py --real             # 强制真模型（需 DEEPSEEK_API_KEY）
  python server.py --mock             # 强制离线模拟，不花钱，用来看界面
  python server.py --port 8080

分层约定（与方案「接口不侵入业务」一致）：
  · 本文件只做 HTTP 语义 → 调用 pipeline / db，不写任何判定或分流逻辑；
  · 长任务不阻塞请求：一次审查会被丢进后台协程，前端拿 task_id 轮询进度；
  · 进度不靠流水线回调，而是从任务快照表统计得出 —— 流水线无感、重启后照样能算。

单进程单 worker：任务注册表在内存里，多 worker 会各存一份进度。要横向扩容时
应把 _TASKS 换成 Redis，接口形状不用变。
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import db
import metrics
import parsing
import pipeline
from llm import ChatResult, LLMClient, TokenBucket

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ==================================================================== 模型
_limiter = TokenBucket()          # 全链路共享一个令牌桶，前端并发点多少次都限得住
_llm_instance = None


def get_llm():
    """懒加载单例：真模型客户端建一次就复用，避免每个任务新建连接池。"""
    global _llm_instance
    if _llm_instance is None:
        if config.MOCK_LLM:
            logging.warning("[server] 离线模拟模式：不调用真实模型，结论仅用于演示流程")
            _llm_instance = MockLLM(_limiter)
        else:
            _llm_instance = LLMClient(config.api_key(), limiter=_limiter)
    return _llm_instance


# ---------------------------------------------------------------- 离线模拟模型
_HIGH_MARKERS = ("概不负责", "不承担", "放弃", "无权", "自行承担", "没收", "无条件", "不予")
_MID_MARKERS = ("违约金", "赔偿", "逾期", "解除", "免责", "视为", "单方", "不得", "扣除")


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。；;])|\n", text or "")
    return [p.strip() for p in parts if p.strip()]


def _mock_judge(term: str) -> Dict[str, Any]:
    """给条款找一句带风险词的原文当证据，让下游证据校验能真的定位到。

    模拟器只保证「输出形状对」，不保证判断质量 —— 所以它有明显的规则痕迹。
    """
    hit, marker = "", ""
    for sent in _sentences(term):
        for mk in _HIGH_MARKERS + _MID_MARKERS:
            if mk in sent:
                hit, marker = sent, mk
                break
        if hit:
            break

    if not hit:
        return {"action": "final", "has_risk": False, "risk_level": "无",
                "reason": "未发现明显权利义务失衡表述（模拟结论）。", "basis": "", "evidence": ""}

    level = "高" if any(k in hit for k in _HIGH_MARKERS) else "中"
    basis = ("《民法典》第四百九十六条 格式条款的提示说明义务"
             if any(k in hit for k in ("免责", "概不负责", "不承担"))
             else "《民法典》第五百七十七条 违约责任")
    return {"action": "final", "has_risk": True, "risk_level": level,
            "reason": f"条款以「{marker}」方式转移或排除我方权利，存在实质减损风险（模拟结论）。",
            "basis": basis, "evidence": hit}


def _mock_review(prompt: str) -> Dict[str, Any]:
    """按 prompt 指纹给出确定性动作，让界面能同时看到采信/修正/驳回三种状态。"""
    h = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)
    if h % 11 == 0:
        return {"action": "reject",
                "reason": "该表述属于义务对等的行业惯例，未实际减损我方权利（模拟结论）。"}
    if h % 7 == 0:
        m = re.search(r"等级：([高中低])", prompt)
        return {"action": "revise", "risk_level": m.group(1) if m else "中",
                "reason": "方向正确，但未计入全局约定的补充，风险描述应收窄（模拟结论）。"}
    return {"action": "accept"}


def _mock_reply(messages: List[Dict[str, str]]) -> str:
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    first = users[0] if users else ""
    joined = "".join(users)

    if "语义分析器" in first:                       # 解析边界：只回一个标签
        cur = first.split("当前候选片段：【", 1)[-1].split("】", 1)[0] if "【" in first else first
        body = cur.strip()
        if len(body) < 20:
            return "ignore"
        if re.match(r"^[一二三四五六七八九十]+[、．.]", body) and len(body) < 30:
            return "merge"
        return "valid"

    if "请开始判断。" in joined:                     # 判定
        m = re.search(r"当前条款（term_id=.*?）：\n(.*?)\n\n.*?请开始判断。", first, re.S)
        return json.dumps(_mock_judge(m.group(1) if m else ""), ensure_ascii=False)

    if "请给出处置决定。" in joined:                 # 复核
        return json.dumps(_mock_review(first), ensure_ascii=False)

    return json.dumps({"action": "final", "has_risk": False, "risk_level": "无",
                       "reason": "模拟兜底", "basis": "", "evidence": ""},
                      ensure_ascii=False)


class MockLLM:
    """离线模拟模型：形状与 LLMClient 一致（chat → ChatResult），不发网络请求。"""

    def __init__(self, limiter=None, delay: float = 0.02):
        self.limiter = limiter
        self.delay = delay
        self.calls = 0

    async def chat(self, messages: List[Dict[str, str]]) -> ChatResult:
        if self.limiter is not None:
            await self.limiter.acquire()
        await asyncio.sleep(self.delay)
        text = _mock_reply(messages)
        self.calls += 1
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return ChatResult(text, max(1, prompt_chars // 2), max(1, len(text) // 2))


# ==================================================================== 合同源
def _safe_id(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    stem = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", stem).strip("_")
    return (stem or "contract")[:60]


def _upload_meta_path(cid: str) -> str:
    return os.path.join(config.UPLOAD_DIR, cid + ".json")


def _upload_display_name(cid: str, fallback: str) -> str:
    path = _upload_meta_path(cid)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("name") or fallback
        except (OSError, ValueError):
            pass
    return fallback


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _sources() -> List[Dict[str, Any]]:
    """样例库 + 上传库，统一成 {contract_id, display_name, path, source}。"""
    out: List[Dict[str, Any]] = []
    if os.path.isdir(config.SAMPLE_CONTRACT_DIR):
        for fn in sorted(os.listdir(config.SAMPLE_CONTRACT_DIR)):
            if fn.endswith(".txt"):
                out.append({"contract_id": os.path.splitext(fn)[0], "display_name": fn,
                            "path": os.path.join(config.SAMPLE_CONTRACT_DIR, fn),
                            "source": "sample"})
    if os.path.isdir(config.UPLOAD_DIR):
        for fn in sorted(os.listdir(config.UPLOAD_DIR)):
            if fn.endswith(".txt"):
                cid = os.path.splitext(fn)[0]
                out.append({"contract_id": cid,
                            "display_name": _upload_display_name(cid, fn),
                            "path": os.path.join(config.UPLOAD_DIR, fn),
                            "source": "upload"})
    return out


def _find_source(contract_id: str) -> Optional[Dict[str, Any]]:
    for s in _sources():
        if s["contract_id"] == contract_id:
            return s
    return None


def _read_source(contract_id: str) -> Tuple[str, str]:
    src = _find_source(contract_id)
    if src is None:
        raise FileNotFoundError(f"找不到合同 {contract_id} 的原文")
    return _read_text(src["path"]), f"{contract_id}.txt"


def _extract_upload_text(filename: str, blob: bytes) -> str:
    """上传只认纯文本与 docx；PDF 不做解析，宁可拒绝也不给假结果。"""
    low = (filename or "").lower()
    if low.endswith(".docx"):
        try:
            import docx                                    # 可选依赖
        except ImportError:
            raise HTTPException(400, "服务端未安装 python-docx，无法解析 .docx，请改为粘贴文本")
        import io
        doc = docx.Document(io.BytesIO(blob))
        return "\n".join(p.text for p in doc.paragraphs)
    if low.endswith(".pdf"):
        raise HTTPException(400, "暂不支持 PDF，请先转成 .txt/.docx 或直接粘贴正文")
    for enc in ("utf-8", "gb18030"):
        try:
            return blob.decode(enc)
        except UnicodeDecodeError:
            continue
    return blob.decode("utf-8", errors="replace")


def _store_upload(display_name: str, text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if len(text) < 50:
        raise HTTPException(400, "正文太短（少于 50 字），不像是完整合同")
    if len(text) > config.MAX_CONTRACT_CHARS:
        raise HTTPException(400, f"正文字数超过上限 {config.MAX_CONTRACT_CHARS}")

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    taken = {s["contract_id"] for s in _sources()}
    cid = _safe_id(display_name)
    if cid in taken:                                        # 同名不同文 → 加内容指纹，绝不覆盖既有合同
        cid = f"{cid}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:6]}"

    with open(os.path.join(config.UPLOAD_DIR, cid + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(_upload_meta_path(cid), "w", encoding="utf-8") as f:
        json.dump({"name": display_name}, f, ensure_ascii=False)
    return {"contract_id": cid, "file_name": display_name, "chars": len(text)}


# ==================================================================== 任务注册表
@dataclass
class TaskRecord:
    task_id: str
    contract_id: str
    mode: str                       # full / parse
    fresh: bool
    status: str = "running"         # running / done / failed
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


_TASKS: Dict[str, TaskRecord] = {}
_ACTIVE: Dict[str, str] = {}        # contract_id -> 正在跑的任务，天然去重
_BG: set = set()


def _progress(rec: TaskRecord) -> Dict[str, Any]:
    """从快照表反推进度：解析 → 判定 → 复核，三个阶段各自有可观测的落库痕迹。"""
    cid = rec.contract_id
    total = len(db.load_terms(cid))
    judge_rows = db.list_judge_snapshots(cid)
    j_done = sum(1 for r in judge_rows if r["status"] == db.FINISHED)
    ev_rows = db.list_evidence(cid)
    review_rows = db.list_review_snapshots(cid)
    r_done = sum(1 for r in review_rows if r["status"] == db.FINISHED)

    if total == 0:
        return {"stage": "解析条款", "terms_total": 0, "judge_done": 0,
                "evidence_total": 0, "review_done": 0, "percent": 4}
    if not ev_rows and not review_rows and j_done < total:
        return {"stage": f"并行判定 {j_done}/{total}", "terms_total": total,
                "judge_done": j_done, "evidence_total": 0, "review_done": 0,
                "percent": min(64, 5 + int(58 * min(j_done, total) / total))}

    denom = max(1, len(ev_rows))
    return {"stage": f"证据校验与复核 {r_done}/{denom}", "terms_total": total,
            "judge_done": j_done, "evidence_total": len(ev_rows), "review_done": r_done,
            "percent": min(97, 66 + int(31 * min(r_done, denom) / denom))}


def _task_view(rec: TaskRecord) -> Dict[str, Any]:
    elapsed = round((rec.finished_at or time.time()) - rec.started_at, 1)
    if rec.status == "running":
        prog = _progress(rec)
    else:
        prog = {"stage": "已完成" if rec.status == "done" else "已失败",
                "percent": 100, "terms_total": (rec.summary or {}).get("terms_total", 0),
                "judge_done": 0, "evidence_total": 0, "review_done": 0}
    return {"task_id": rec.task_id, "contract_id": rec.contract_id, "mode": rec.mode,
            "fresh": rec.fresh, "status": rec.status, "elapsed_s": elapsed,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.started_at)),
            "error": rec.error, "summary": rec.summary, "progress": prog}


def _active_task(contract_id: str) -> Optional[Dict[str, Any]]:
    tid = _ACTIVE.get(contract_id)
    if not tid:
        return None
    rec = _TASKS.get(tid)
    return _task_view(rec) if rec else None


async def _execute(rec: TaskRecord) -> None:
    """后台跑一次任务。异常一律落到记录里，绝不吞成 500。"""
    try:
        raw, pipeline_name = _read_source(rec.contract_id)
        if rec.fresh:
            db.reset_contract(rec.contract_id)

        if rec.mode == "parse":
            parsed = await parsing.parse_contract(
                rec.contract_id, pipeline_name, raw, llm=None, use_llm=False, persist=True)
            result = {"contract_id": rec.contract_id, "file_name": pipeline_name,
                      "mode": "parse", "terms_total": len(parsed["terms"]),
                      "risk_total": 0, "needs_human": 0, "final": [],
                      "duration_s": round(time.time() - rec.started_at, 2)}
            report = {}
            db.save_result(rec.contract_id, rec.task_id, "done", result, report,
                           result["duration_s"])
            rec.summary = {"terms_total": result["terms_total"], "risk_total": 0,
                           "needs_human": 0, "duration_s": result["duration_s"]}
        else:
            out = await pipeline.review_contract(
                rec.contract_id, pipeline_name, raw, llm=get_llm(),
                persist=True, resume=not rec.fresh, use_llm_parse=True)
            ctx = db.load_global_ctx(rec.contract_id)
            report = metrics.build_report(out, ctx["clean_text"] if ctx else "")
            db.save_result(rec.contract_id, rec.task_id, "done", out, report,
                           out.get("duration_s", 0))
            rec.summary = {"terms_total": out["terms_total"], "risk_total": out["risk_total"],
                           "needs_human": out["needs_human"], "duration_s": out["duration_s"],
                           "llm_calls": (out.get("scheduler") or {}).get("llm_calls", 0)}

        rec.status = "done"
        logging.info(f"[server] 任务 {rec.task_id} 完成：{rec.summary}")
    except Exception as e:                                   # noqa: BLE001 兜底必须留下痕迹
        rec.status = "failed"
        rec.error = f"{type(e).__name__}: {e}"
        logging.exception(f"[server] 任务 {rec.task_id} 失败")
        try:
            db.save_result(rec.contract_id, rec.task_id, "failed",
                           duration_s=round(time.time() - rec.started_at, 2), error=rec.error)
        except Exception:
            logging.exception("[server] 失败结果归档也没写进去")
    finally:
        rec.finished_at = time.time()
        _ACTIVE.pop(rec.contract_id, None)


def _spawn(rec: TaskRecord) -> None:
    _TASKS[rec.task_id] = rec
    _ACTIVE[rec.contract_id] = rec.task_id
    task = asyncio.create_task(_execute(rec))
    _BG.add(task)
    task.add_done_callback(_BG.discard)
    if len(_TASKS) > 200:                                    # 内存里只留最近的记录
        for tid in [t for t, r in _TASKS.items() if r.status != "running"][:100]:
            _TASKS.pop(tid, None)


# ==================================================================== 视图组装
def _term_views(contract_id: str, terms: List[Dict[str, Any]],
                finals: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把四张表按 term_id 合到一行 —— 前端只吃这一个结构，不做多表 join。"""
    judge = {r["term_id"]: r for r in db.list_judge_snapshots(contract_id)}
    evid = {r["term_id"]: r for r in db.list_evidence(contract_id)}
    reviews = {r["term_id"]: r for r in db.list_review_snapshots(contract_id)}

    out = []
    for t in terms:
        tid = t["term_id"]
        j, e, rv = judge.get(tid), evid.get(tid), reviews.get(tid)
        out.append({
            "term_id": tid, "term_no": t["term_no"], "term_text": t["term_text"],
            "complexity": t["complexity"], "char_start": t["char_start"],
            "char_end": t["char_end"],
            "judge": None if not j else {
                "status": j["status"], "rounds": j["cur_round"],
                "max_rounds": j["max_rounds"], "tokens": j["token_usage"],
                "tool_calls": j.get("tool_calls") or [], "risk": j.get("risk_draft"),
            },
            "evidence": None if not e else {
                "verify_status": e["verify_status"], "similarity": e["similarity"],
                "start": e["start_offset"], "end": e["end_offset"],
                "matched_text": e["matched_text"],
            },
            "review": None if not rv else {
                "status": rv["status"], "action": rv["action"], "rounds": rv["cur_round"],
                "tokens": rv["token_usage"], "final": rv.get("final_risk"),
            },
            "final": finals.get(tid) or (rv.get("final_risk") if rv else None),
        })
    return out


def _detail(contract_id: str) -> Dict[str, Any]:
    src = _find_source(contract_id)
    ctx = db.load_global_ctx(contract_id)
    latest = db.load_latest_result(contract_id)

    if ctx is None:
        if src is None:
            raise HTTPException(404, f"合同 {contract_id} 不存在")
        raw = _read_text(src["path"])
        return {"contract_id": contract_id, "file_name": src["display_name"],
                "source": src["source"], "parsed": False,
                "clean_text": parsing.clean_text(raw), "raw_chars": len(raw),
                "meta": {}, "global_rules": [], "custom_terms": {},
                "terms": [], "result": None, "report": None,
                "last_run": None, "running": _active_task(contract_id)}

    terms = db.load_terms(contract_id)
    result = (latest or {}).get("result") or {}
    finals = {f.get("term_id"): f for f in (result.get("final") or []) if f.get("term_id")}

    return {
        "contract_id": contract_id,
        "file_name": (latest and (result.get("file_name"))) or ctx.get("file_name") or contract_id,
        "source": src["source"] if src else "db",
        "parsed": True,
        "clean_text": ctx["clean_text"],
        "raw_chars": len(ctx["clean_text"]),
        "meta": ctx["meta"], "global_rules": ctx["global_rules"],
        "custom_terms": ctx["custom_terms"],
        "terms": _term_views(contract_id, terms, finals),
        "result": result or None,
        "report": (latest or {}).get("report") or None,
        "last_run": None if not latest else {
            "run_id": latest["run_id"], "status": latest["status"],
            "created_at": latest["created_at"], "error": latest["error"],
            "duration_s": latest["duration_s"],
        },
        "running": _active_task(contract_id),
    }


# ==================================================================== 请求模型
class ReviewRequest(BaseModel):
    fresh: bool = False          # True = 丢弃已有快照从头跑
    parse_only: bool = False     # True = 只切条款，不调模型判定


class TextRequest(BaseModel):
    name: str = "粘贴的合同.txt"
    text: str


# ==================================================================== 应用
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    db.init_db()
    mode = "离线模拟" if config.MOCK_LLM else f"真实模型 {config.LLM_MODEL}"
    logging.info(f"[server] 启动完成｜模式：{mode}｜库：{config.DB_PATH}")
    yield


app = FastAPI(title="合同风险审查多 Agent 系统", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "mock": config.MOCK_LLM,
        "model": config.LLM_MODEL if not config.MOCK_LLM else "mock-llm",
        "api_key_set": bool(os.getenv(config.LLM_API_KEY_ENV, "").strip()),
        "db_path": config.DB_PATH,
        "contracts": len(_sources()),
        "running": len(_ACTIVE),
        "limits": {"concurrency": config.MAX_CONCURRENT_TASKS, "qps": config.QPS_LIMIT,
                   "token_per_task": config.TOKEN_LIMIT_PER_TASK},
    }


@app.get("/api/contracts")
async def list_contracts():
    overview = {r["contract_id"]: r for r in db.list_contracts()}
    latest = db.list_latest_results()
    items = []
    for s in _sources():
        cid = s["contract_id"]
        o, l = overview.get(cid) or {}, latest.get(cid) or {}
        try:
            chars = len(_read_text(s["path"]))
        except OSError:
            chars = 0
        items.append({
            "contract_id": cid, "file_name": s["display_name"], "source": s["source"],
            "chars": chars, "parsed": bool(o),
            "terms_total": o.get("terms") or l.get("terms_total") or 0,
            "risk_total": l.get("risk_total", 0), "needs_human": l.get("needs_human", 0),
            "reviewed_at": l.get("created_at"), "review_status": l.get("status"),
            "running": cid in _ACTIVE,
        })
    items.sort(key=lambda x: (x["source"] != "sample", x["file_name"]))
    return {"total": len(items), "items": items}


@app.get("/api/contracts/{contract_id}")
async def get_contract(contract_id: str):
    return _detail(contract_id)


@app.post("/api/contracts/{contract_id}/review")
async def start_review(contract_id: str, body: ReviewRequest):
    if _find_source(contract_id) is None:
        raise HTTPException(404, f"合同 {contract_id} 不存在")
    if not config.MOCK_LLM and not os.getenv(config.LLM_API_KEY_ENV, "").strip():
        raise HTTPException(400, f"未配置环境变量 {config.LLM_API_KEY_ENV}，"
                                 f"无法调用模型；可先用 --mock 启动离线演示")

    if contract_id in _ACTIVE:                       # 同一合同不并跑，避免快照互相踩
        return {"task_id": _ACTIVE[contract_id], "reused": True}

    rec = TaskRecord(task_id=uuid.uuid4().hex[:12], contract_id=contract_id,
                     mode="parse" if body.parse_only else "full", fresh=body.fresh)
    _spawn(rec)
    return {"task_id": rec.task_id, "reused": False}


@app.get("/api/tasks")
async def list_tasks(contract_id: Optional[str] = None, limit: int = 50):
    recs = [r for r in _TASKS.values()
            if contract_id is None or r.contract_id == contract_id]
    recs.sort(key=lambda r: r.started_at, reverse=True)
    return {"items": [_task_view(r) for r in recs[:limit]]}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    rec = _TASKS.get(task_id)
    if rec is None:
        raise HTTPException(404, "任务不存在或已被清理")
    view = _task_view(rec)
    if rec.status != "running":
        latest = db.load_latest_result(rec.contract_id)
        if latest and latest["run_id"] == task_id:
            view["result"] = latest["result"]
            view["report"] = latest["report"]
    return view


@app.post("/api/contracts/upload")
async def upload_contract(file: UploadFile = File(...)):
    blob = await file.read()
    if len(blob) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"文件超过 {config.MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限")
    text = _extract_upload_text(file.filename or "", blob)
    return _store_upload(file.filename or "上传的合同.txt", text)


@app.post("/api/contracts/text")
async def create_from_text(body: TextRequest):
    return _store_upload(body.name or "粘贴的合同.txt", body.text)


@app.delete("/api/contracts/{contract_id}")
async def delete_contract(contract_id: str):
    src = _find_source(contract_id)
    if src is None:
        raise HTTPException(404, "合同不存在")
    if src["source"] != "upload":
        raise HTTPException(400, "样例合同不允许删除")
    if contract_id in _ACTIVE:
        raise HTTPException(409, "该合同正在审查中，请等任务结束")
    for path in (src["path"], _upload_meta_path(contract_id)):
        if os.path.exists(path):
            os.remove(path)
    db.reset_contract(contract_id)
    return {"ok": True}


@app.get("/api/regression")
async def regression(limit: int = 100):
    rows = db.list_regression(limit)
    for r in rows:
        try:
            r["detail"] = json.loads(r["detail"]) if r.get("detail") else {}
        except (TypeError, ValueError):
            r["detail"] = {}
    return {"total": len(rows), "items": rows}


# ---------------------------------------------------------------- 静态前端
if os.path.isdir(config.WEB_DIR):
    app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")


@app.get("/")
async def index():
    page = os.path.join(config.WEB_DIR, "index.html")
    if not os.path.exists(page):
        raise HTTPException(404, "前端未安装：缺少 web/index.html")
    return FileResponse(page)


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(config.WEB_DIR, "favicon.svg"),
                        media_type="image/svg+xml")


# ==================================================================== 启动入口
def main() -> None:
    parser = argparse.ArgumentParser(description="合同风险审查多 Agent 系统 —— Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--real", action="store_true", help="强制调用真实模型")
    parser.add_argument("--mock", action="store_true", help="强制离线模拟模型")
    parser.add_argument("--db", help="覆盖数据库路径")
    args = parser.parse_args()

    if args.db:
        config.DB_PATH = args.db
    if args.real:
        config.MOCK_LLM = False
    elif args.mock:
        config.MOCK_LLM = True
    elif not os.getenv(config.LLM_API_KEY_ENV, "").strip():
        config.MOCK_LLM = True                       # 没密钥就自动降级，别让人卡在 401 上

    import uvicorn
    print("=" * 62)
    print("  合同风险审查多 Agent 系统")
    print(f"  模式：{'离线模拟（结论仅供演示）' if config.MOCK_LLM else config.LLM_MODEL}")
    print(f"  接口文档：http://{args.host}:{args.port}/docs")
    print(f"  界面：    http://{args.host}:{args.port}/")
    print("=" * 62)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
