"""evidence.py —— 证据校验模块（方案 7）。

这是规则化的文本匹配程序，不是 Agent：没有自主思考，只按相似度阈值分流。
职责只有一件事 —— 验证判定结论引用的证据片段能否在原文中定位，
并给出该片段在 clean_full_text 里的 start/end 字符偏移（证据锚点）。

三段式：
  7.2 文本归一化   normalize()：同一套归一键处理原文与证据片段，消除格式差异
  7.3 滑动窗口模糊匹配  find_span()：滑动窗口 + 相似度，取最高分窗口
  7.4 三路分流     verify()：定位成功 / 定位失败 / 缺少有效证据

偏移口径：所有对外返回的 start/end 都是【原始 clean_full_text 的字符下标】，
归一化只是内部手段。所以归一时必须同步维护「归一化下标 → 原始下标」的映射表，
否则定位出来的锚点全是错的（这一层是方案 7.3「取得 start/end 字符偏移」的前提）。
"""
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import config
import db

# 不可见控制字符：直接剔除（剔除会让长度变化，靠映射表还原）
_INVISIBLE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\u2028\u2029\ufeff]")
# 空白（含全角空格与各类换行）：压缩为单个半角空格
_SPACE = re.compile(r"[\s\u3000]+")


# ==================================================================== 7.2 归一化
def normalize(text: str) -> Tuple[str, List[int]]:
    """归一化，同时返回「归一化后每个字符 → 原文下标」的映射。

    规则（方案 7.2）：全角转半角、连续空格与换行压缩为单个空格、统一中英文标点、
    剔除不可见控制字符。逐字符处理，保证 index_map 与归一化文本严格一一对应。
    """
    out_chars: List[str] = []
    index_map: List[int] = []
    prev_space = False

    for i, ch in enumerate(text or ""):
        if _INVISIBLE.match(ch):
            continue                                   # 剔除，不产生输出

        # 全角 → 半角（NFKC 对多数全角字符有效；长度不变或变化时以下面的逐字符判断兜底）
        converted = unicodedata.normalize("NFKC", ch)

        if _SPACE.match(ch):
            if prev_space:                             # 连续空白只留一个
                continue
            out_chars.append(" ")
            index_map.append(i)
            prev_space = True
            continue

        prev_space = False
        if len(converted) == 1:
            out_chars.append(converted)
            index_map.append(i)
        else:
            # NFKC 把一个字符拆成多个（如某些连字）：只取首字符，映射仍指回同一个原文位置
            out_chars.append(converted[0] if converted else ch)
            index_map.append(i)

    # 去掉首尾空格（映射同步裁掉）
    start = 0
    end = len(out_chars)
    while start < end and out_chars[start] == " ":
        start += 1
    while end > start and out_chars[end - 1] == " ":
        end -= 1

    return "".join(out_chars[start:end]), index_map[start:end]


def _to_original(index_map: List[int], s: int, e: int) -> Tuple[int, int]:
    """把归一化区间 [s, e) 换算回原始文本区间 [start, end)。"""
    if not index_map or s >= len(index_map):
        return 0, 0
    e = min(e, len(index_map))
    return index_map[s], index_map[e - 1] + 1


# ==================================================================== 7.3 模糊匹配
def _ratio(a: str, b: str) -> float:
    """相似度：序列比（等价于归一化编辑距离的相似度形式，0~1）。"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _anchor_positions(text: str, evidence: str, limit: int = 40) -> List[int]:
    """用证据里的若干锚点串在原文里找候选起点，避免全量滑窗。

    从长到短试锚点（12/8/6/4 字），命中即用；一条都没命中才退回粗扫。
    这样在「模型复述基本忠实」的常见情况下，成本是几十次相似度计算而不是几万次。
    """
    positions: List[int] = []
    n = len(evidence)
    for anchor_len in (12, 8, 6, 4):
        if anchor_len > n or anchor_len < config.MATCH_MIN_ANCHOR:
            continue
        # 头 / 中 / 尾三处取样，覆盖模型两头截断或中间改写的情况
        starts = {0, max(0, (n - anchor_len) // 2), max(0, n - anchor_len)}
        for off in sorted(starts):
            anchor = evidence[off:off + anchor_len]
            if not anchor.strip():
                continue
            idx = text.find(anchor)
            while idx != -1 and len(positions) < limit:
                positions.append(max(0, idx - off))    # 换算成窗口起点
                idx = text.find(anchor, idx + 1)
        if positions:
            break
    return positions


def _score_window(text: str, start: int, length: int, evidence: str) -> float:
    """在 start 处取长度约为 length 的窗口打分；长度做 ±20% 抖动，避免切在半个词上。"""
    best = 0.0
    n = len(text)
    for factor in (1.0, 0.9, 1.1, 0.8, 1.2):
        w = max(4, int(length * factor))
        if start + w > n:
            w = n - start
        if w <= 0:
            continue
        best = max(best, _ratio(evidence, text[start:start + w]))
    return best


def find_span(clean_text: str, evidence: str,
              threshold: Optional[float] = None) -> Dict[str, object]:
    """滑动窗口模糊匹配：返回最优窗口的相似度与原始偏移。

    流程：先精确匹配（最快、也最常见），失败再锚点定位，最后才粗扫 + 局部细扫。
    """
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    ev_raw = (evidence or "").strip()
    if not ev_raw:
        return {"similarity": 0.0, "start": None, "end": None, "matched": ""}

    text_norm, index_map = normalize(clean_text)
    ev_norm, _ = normalize(ev_raw)
    if not text_norm or not ev_norm:
        return {"similarity": 0.0, "start": None, "end": None, "matched": ""}

    # 1) 精确命中：模型逐字复制时走这条，相似度记满分
    pos = text_norm.find(ev_norm)
    if pos != -1:
        s, e = _to_original(index_map, pos, pos + len(ev_norm))
        return {"similarity": 1.0, "start": s, "end": e, "matched": (clean_text or "")[s:e]}

    n = len(text_norm)
    ev_len = len(ev_norm)
    best = {"similarity": 0.0, "start": None, "end": None, "matched": ""}

    def _consider(win_start: int) -> None:
        if win_start < 0 or win_start >= n:
            return
        sim = _score_window(text_norm, win_start, ev_len, ev_norm)
        if sim > best["similarity"]:
            w = min(ev_len, n - win_start)
            s, e = _to_original(index_map, win_start, win_start + w)
            best.update({"similarity": round(sim, 4), "start": s, "end": e,
                         "matched": (clean_text or "")[s:e]})

    # 2) 锚点定位：只看候选位置，成本可控
    for p in _anchor_positions(text_norm, ev_norm):
        for delta in range(-config.MATCH_REFINE_RADIUS, config.MATCH_REFINE_RADIUS + 1):
            _consider(p + delta)
        if best["similarity"] >= threshold:
            return best

    # 3) 兜底粗扫：锚点全落空（模型大幅改写）时，才逐段扫一遍
    if best["similarity"] < threshold:
        step = config.MATCH_STEP_COARSE
        for win_start in range(0, max(1, n - ev_len + 1), step):
            # 粗扫只做低成本预筛，避免每次滑窗都跑序列比
            if _quick_overlap(text_norm[win_start:win_start + ev_len], ev_norm) < 0.2:
                continue
            _consider(win_start)
            if best["similarity"] >= threshold:
                return best

    return best


def _quick_overlap(a: str, b: str) -> float:
    """字符 2-gram 覆盖率，纯预筛用，不参与最终相似度。"""
    def grams(s: str) -> set:
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga)


# ==================================================================== 7.4 三路分流
def verify(clean_text: str, evidence: str,
           threshold: Optional[float] = None) -> Dict[str, object]:
    """证据校验入口：返回 {verify_status, start, end, similarity, matched_text}。

    三态判定（方案表 5）：
      · 相似度 ≥ 阈值            → 定位成功，带锚点进复核
      · 0 < 相似度 < 阈值        → 定位失败，降级标记待人工确认后进复核
      · 相似度为 0 或证据为空    → 缺少有效证据，判无效并纳入回归池
    """
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold

    if not (evidence or "").strip():
        return {"verify_status": db.EVIDENCE_NONE, "start": None, "end": None,
                "similarity": 0.0, "matched_text": ""}

    hit = find_span(clean_text, evidence, threshold)
    sim = float(hit["similarity"])

    if sim >= threshold:
        status = db.EVIDENCE_OK
    elif sim > 0:
        status = db.EVIDENCE_FAIL
    else:
        status = db.EVIDENCE_NONE

    return {
        "verify_status": status,
        "start": hit["start"],
        "end": hit["end"],
        "similarity": sim,
        "matched_text": hit["matched"] or "",
    }
