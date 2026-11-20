"""parsing.py —— 解析 Agent（方案 5）。

六步：预处理 → 正则锚点粗切 → LLM 语义边界判定 → 条款组装 → 双路输出 → 复杂度打标。

设计要点（方案 5.2）：正则只负责找「疑似条款的起点」，不判断片段是不是独立条款，
也不做任何语义过滤；valid / merge / ignore 的判断全部交给 LLM。
这是解析 Agent 区别于纯脚本方案的核心特征，也是本文件唯一允许调 LLM 的地方。

偏移口径：所有 char_start / char_end 都是【clean_full_text 中的字符下标】，
clean_full_text 由本模块产出并落库，后续证据校验直接以它为坐标系。
"""
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import config
import db
import prompts
from llm import LLMClient

# ---------------------------------------------------------------- 清洗用正则
_HDR_LINE = re.compile(r"^[ \t\u3000]*[-=_—－·*]{3,}[ \t\u3000]*$")      # 页眉分割线
_PAGE_NO = re.compile(r"^[ \t\u3000]*(第\s*\d+\s*页(\s*/?\s*共\s*\d+\s*页)?|\d{1,3})[ \t\u3000]*$")
_TRAILING = re.compile(r"[ \t\u3000]+$")
_BLANK_RUN = re.compile(r"\n{3,}")

_PARTY = re.compile(r"(甲\s*方|乙\s*方)[（(][^）)]{0,12}[）)]?\s*[：:]\s*([^\n，,。;；]{2,40})")
_SUBJECT = re.compile(r"(?:标的物?名称|产品名称|服务内容|标的)\s*[：:]\s*([^\n。]{2,60})")
_ACCEPT = re.compile(r"(?:验收标准|质量标准)\s*[：:]\s*([^\n。]{2,90})")
_DEFINE = re.compile(r"本合同所称[“\"「]?([^”\"」，,。：:]{1,20})[”\"」]?\s*(?:，|,)?\s*(?:包括|是指|指)\s*([^。；\n]{2,90})")
_GLOBAL_RULE = re.compile(r"^.*(一律|均须|均应|未经.{0,10}不得|本合同所有|双方确认).*$")


# ==================================================================== 5.1 预处理
def clean_text(raw: str) -> str:
    """正则轻量清洗：统一换行、去页眉分割线与页码行、压空行、去行尾空格。

    不做极致容错 —— 方案 1.2 明确只适配规范商事合同，退化文档应在上游人工预处理。
    """
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    kept: List[str] = []
    for line in text.split("\n"):
        if _HDR_LINE.match(line):
            continue
        if _PAGE_NO.match(line):
            continue
        kept.append(_TRAILING.sub("", line))
    out = "\n".join(kept)
    out = _BLANK_RUN.sub("\n\n", out)
    return out.strip()


# ==================================================================== 5.2 正则锚点粗切
def split_candidates(text: str) -> List[Tuple[int, int, str]]:
    """按正则锚点把全文切成候选片段，返回 [(start, end, 片段)]。

    锚点之前的「前言区」也作为一个候选（通常会判成 ignore），
    否则合同开头的当事人信息、鉴于条款会被整段丢掉。
    """
    anchors = [m.start() for m in re.finditer(config.ANCHOR_PATTERN, text)]
    if not anchors:
        return [(0, len(text), text)] if text.strip() else []

    spans: List[Tuple[int, int]] = []
    if anchors[0] > 0:
        spans.append((0, anchors[0]))
    for i, pos in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else len(text)
        spans.append((pos, end))

    out = []
    for s, e in spans:
        piece = text[s:e]
        if len(piece.strip()) < config.MIN_CANDIDATE_CHARS:
            continue
        out.append((s, e, piece))
    return out


# ==================================================================== 5.3 语义边界判定
def parse_label(resp: str) -> str:
    """把模型回复收敛成 valid / merge / ignore 之一。

    模型偶尔会带解释或加粗，这里只认第一个出现的标签词，认不出按 ignore 处理
    （宁可丢一条可疑片段，也不要把噪音组装成条款 —— 后者会污染后续全部判定）。
    """
    low = (resp or "").strip().lower()
    for label in prompts.PARSE_LABELS:
        if label in low:
            return label
    return "ignore"


async def judge_boundary(llm: LLMClient, prev_text: str, current_text: str) -> str:
    """单条候选片段的边界判定，Prompt 固定，只输出一个标签。"""
    prompt = prompts.PARSE_BOUNDARY.format(
        prev_text=prev_text[-200:] or "（无）",
        current_text=current_text[:600],
    )
    res = await llm.chat([{"role": "user", "content": prompt}])
    return parse_label(res.text)


def heuristic_boundary(prev_text: str, current_text: str) -> str:
    """离线兜底：不调 LLM 时的确定性判定，只用于自测与断网跑通链路。

    规则很粗：带编号 + 有实质动词 → valid；纯标题（短且无标点结尾）→ merge/ignore。
    线上永远走 judge_boundary，这个函数不进生产路径。
    """
    body = current_text.strip()
    if len(body) < config.MIN_CANDIDATE_CHARS:
        return "ignore"
    if re.match(r"^[一二三四五六七八九十]+[、．.]", body) and len(body) < 30:
        return "merge"          # 一级标题，并入下一段
    return "valid"


# ==================================================================== 5.4 组装
def assemble(candidates: List[Tuple[int, int, str]],
             labels: List[str]) -> List[Dict[str, Any]]:
    """按标签做确定性组装：valid 新起一条，merge 并到上一条末尾，ignore 丢弃。

    三种分支都要维护「上文片段」，保证下一轮判定始终有语境（方案 5.4）。
    """
    terms: List[Dict[str, Any]] = []
    prev_text = ""
    for (s, e, piece), label in zip(candidates, labels):
        if label == "valid":
            terms.append({"start": s, "end": e, "text": piece.strip()})
        elif label == "merge":
            if terms:                                  # 有上一条才合并，否则等同 valid
                last = terms[-1]
                last["end"] = e
                last["text"] = f"{last['text']}\n{piece.strip()}"
            else:
                terms.append({"start": s, "end": e, "text": piece.strip()})
        # ignore：只更新上文，不入列表
        prev_text = piece
    return [t for t in terms if len(t["text"]) >= config.MIN_TERM_CHARS]


# ==================================================================== 复杂度打标
def tag_complexity(term_text: str) -> str:
    """简单 / 复杂标签（方案 6.4）：决定判定任务的动态轮次上限。

    判据是「是否需要多轮核验」：够长、带交叉引用词、带金额比例，都算复杂。
    这一步是脚本判断，不劳 LLM。
    """
    text = term_text or ""
    if len(text) >= config.COMPLEX_MIN_CHARS:
        return "complex"
    hits = sum(1 for m in config.COMPLEX_MARKERS if m in text)
    if hits >= 2:
        return "complex"
    if re.search(r"\d+(\.\d+)?\s*%|第[一二三四五六七八九十百\d]+条|附件", text):
        return "complex"
    return "simple"


# ==================================================================== 5.5 双路输出
def extract_global_ctx(text: str) -> Dict[str, Any]:
    """第一路输出：从全文抽全局共享上下文。方案 5.5 说「采用简化实现，不构建复杂引用图」，
    所以这里是正则抽取 + 关键词兜底，不上 LLM。"""
    meta: Dict[str, str] = {}
    for m in _PARTY.finditer(text):
        role = m.group(1).replace(" ", "")
        val = m.group(2).strip(" 　:：")
        if role and val and role not in meta:
            meta[role] = val
    subj = _SUBJECT.search(text) or re.search(r"向乙方(?:采购|订购|购买)([^\n，,。；]{2,30}?)(?:事宜|，|。|；|$)", text)
    if subj:
        meta["标的"] = subj.group(1).strip()
    acc = _ACCEPT.search(text)
    if acc:
        meta["验收标准"] = acc.group(1).strip()

    custom_terms: Dict[str, str] = {}
    for m in _DEFINE.finditer(text):
        name = m.group(1).strip()
        if name and name not in custom_terms:
            custom_terms[name] = m.group(2).strip()[:80]

    rules: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if 10 <= len(s) <= 120 and _GLOBAL_RULE.match(s) and s not in rules:
            rules.append(s)
        if len(rules) >= 5:
            break

    return {"meta": meta, "custom_terms": custom_terms, "global_rules": rules}


# ==================================================================== 主流程
async def parse_contract(contract_id: str, file_name: str, raw_text: str,
                         llm: Optional[LLMClient] = None,
                         use_llm: bool = True,
                         persist: bool = True) -> Dict[str, Any]:
    """解析入口：一份合同 → (全局上下文, 条款列表)，两路都落库。

    llm 为 None 或 use_llm=False 时走启发式边界判定（离线/自测路径）。
    返回 {"clean_text", "global_ctx", "terms"}，供调度器直接 Fan-out。
    """
    text = clean_text(raw_text)
    candidates = split_candidates(text)
    logging.info(f"[parse] {file_name} 清洗后 {len(text)} 字，正则锚点切出 {len(candidates)} 个候选片段")

    labels: List[str] = []
    if use_llm and llm is not None:
        prev = ""
        for idx, (_s, _e, piece) in enumerate(candidates, start=1):
            label = await judge_boundary(llm, prev, piece)
            labels.append(label)
            prev = piece
            logging.debug(f"[parse] 候选 {idx}/{len(candidates)} → {label}")
    else:
        prev = ""
        for _s, _e, piece in candidates:
            labels.append(heuristic_boundary(prev, piece))
            prev = piece

    assembled = assemble(candidates, labels)

    ctx = extract_global_ctx(text)
    terms: List[Dict[str, Any]] = []
    for i, t in enumerate(assembled, start=1):
        terms.append({
            "term_id": f"{contract_id}-T{i:03d}",
            "term_no": i,
            "term_text": t["text"],
            "complexity": tag_complexity(t["text"]),
            "char_start": t["start"],
            "char_end": t["end"],
        })
    term_index = {t["term_id"]: t["term_text"][:40] for t in terms}

    if persist:
        db.save_global_ctx(contract_id, file_name, text, ctx["meta"],
                           ctx["global_rules"], ctx["custom_terms"], term_index)
        db.save_terms(contract_id, terms)

    n_complex = sum(1 for t in terms if t["complexity"] == "complex")
    logging.info(f"[parse] {file_name} 组装出 {len(terms)} 条条款（复杂 {n_complex} 条）")

    return {"clean_text": text, "global_ctx": ctx, "terms": terms}
