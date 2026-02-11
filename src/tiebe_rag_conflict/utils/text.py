# utils/text.py
from __future__ import annotations
import re
from typing import Iterable, Optional
from collections import Counter

__all__ = [
    "normalize_text",
    "split_sentences",
    "strip_answer_sentences",
    "classify",   # 新增
]

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)

def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    tokens = _WORD_RE.findall(text.lower())
    return "".join(tokens)

def split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter based on punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def strip_answer_sentences(sentences: Iterable[str], answer: str) -> list[str]:
    answer_norm = normalize_text(answer)
    kept: list[str] = []
    for sent in sentences:
        if answer_norm and answer_norm in normalize_text(sent):
            continue
        kept.append(sent)
    return kept

# ---------------------------------------------------------------------
# 以下是分類用工具函式
# ---------------------------------------------------------------------

def _tokenize(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _multiset_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    inter = sum(min(ca[t], cb[t]) for t in set(ca) | set(cb))
    union = sum(max(ca[t], cb[t]) for t in set(ca) | set(cb))
    return inter / union if union else 0.0

# --- 數字一致性（避免數值不一致卻被判等價） ---
_NUM_RE = re.compile(r"""
    (?P<prefix>[$€£])?                         # 可選貨幣符號
    (?P<num>\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?  # 整數或小數，支援千分位
    (?P<suffix>\s*%|[kKmMbB])?                 # 可選百分比或量級 (k,m,b)
""", re.VERBOSE)

def _extract_numbers(text: str) -> list[str]:
    """抓出文本中的顯著數字（含 $, %, k/m/b 量級），回傳標準化後的字串清單。"""
    if not text:
        return []
    out: list[str] = []
    for m in _NUM_RE.finditer(text):
        prefix = (m.group("prefix") or "").strip()
        raw = m.group("num")
        suffix = (m.group("suffix") or "").strip().lower()
        core = raw.replace(",", "")  # 去逗號，保留小數點
        tag = ""
        if suffix in {"k", "m", "b"}:
            tag = suffix
            suffix = ""
        norm = f"{prefix}{core}{suffix}{tag}"  # 例: $44b, 10.1%, 134
        out.append(norm)
    return out

def _numbers_consistent(a: str, b: str) -> bool:
    """
    若雙方皆含數字，則要求數字集合相等（順序不重要）。
    若其中一方沒有數字，回傳 True（不以數字否決）。
    """
    nums_a = set(_extract_numbers(a))
    nums_b = set(_extract_numbers(b))
    if nums_a and nums_b:
        return nums_a == nums_b
    return True

# --- Reject 規則（涵蓋各種拒答等價表述） ---
_REJECT_PATTERNS = re.compile(
    "|".join([
        r"\bnot\s*[_-]?\s*supported\b",
        r"\bnot\s+enough\s+information\b",
        r"\bno\s+information\b",
        r"\bcannot\s+answer\b",
        r"\bcan(?:not|'t)\s+determine\b",
        r"\binsufficient\s+context\b",
        r"\bcontext\s+does\s+not\s+(?:mention|contain)\b",
        r"\bunknown\b",
        r"\bi\s+don'?t\s+know\b",
        r"\bno\s+data\b",
    ]),
    re.IGNORECASE,
)

def _is_reject(pred: Optional[str]) -> bool:
    if pred is None:
        return True
    p = str(pred).strip()
    if not p:
        return True
    # 把常見包裹符號移除（quote / code fence）
    p = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", p).strip()
    p = re.sub(r"^['\"]|['\"]$", "", p).strip()
    return bool(_REJECT_PATTERNS.search(p))

# --- 語義等價（含子字串、Jaccard、multiset overlap 與數字一致性） ---
def _sem_equal(a: Optional[str], b: Optional[str],
               allow_substring: bool = True,
               jaccard_thr: float = 0.75,
               overlap_thr: float = 0.80) -> bool:
    if not a or not b:
        return False
    if not _numbers_consistent(a, b):
        return False

    na, nb = normalize_text(a), normalize_text(b)
    if na == nb:
        return True

    if allow_substring:
        if len(na) >= 3 and na in nb:
            return True
        if len(nb) >= 3 and nb in na:
            return True

    ta, tb = _tokenize(a), _tokenize(b)
    if _jaccard(set(ta), set(tb)) >= jaccard_thr:
        return True
    if _multiset_overlap(ta, tb) >= overlap_thr:
        return True
    return False

# --- 主分類器 ---
def classify(pred: Optional[str], gold: Optional[str], false: Optional[str]) -> str:
    """
    Returns one of: "Reject", "Stick", "Follow"
    - Reject: pred 含拒答語（含 not supported 等變體）或為空
    - Stick:  pred 與 gold 語義近似（含數字一致）
    - Follow: pred 與 false 語義近似；或其他非拒答但未等於 gold 的答案
    """
    if _is_reject(pred):
        return "Reject"
    if _sem_equal(pred, gold):
        return "Stick"
    if _sem_equal(pred, false):
        return "Follow"
    return "Follow"
