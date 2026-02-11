from __future__ import annotations

from collections import Counter
from typing import Iterable

from tiebe_rag_conflict.utils.text import normalize_text


import re

from tiebe_rag_conflict.utils.text import normalize_text
from tiebe_rag_conflict.utils.text import classify as _classify  # ✅ 單一來源

def normalize(s: str | None) -> str:
    return normalize_text(s)

def classify(pred: str, gold: str, false: str | None) -> str:
    # 直接用 utils/text.py 的實作，已包含：
    # - "not supported" / "not enough information" 等拒答偵測
    # - 語義相等（Stick）
    # - 語義接近 false（Follow）
    return _classify(pred, gold, false)



def compute_csi(rows: Iterable[dict]) -> dict[str, float | int]:
    counter = Counter()
    for row in rows:
        pred = row.get("pred", "")
        gold = row.get("gold", "")
        false = row.get("false")
        label = classify(pred, gold, false)
        counter[label] += 1
    total = counter["Stick"] + counter["Follow"] + counter["Reject"]
    if total == 0:
        return {"CSI": 0.0, "Acc": 0.0, "Stick": 0, "Follow": 0, "Reject": 0, "N": 0}
    csi = (counter["Stick"] + counter["Reject"]) / total
    acc = counter["Stick"] / total
    return {
        "CSI": csi,
        "Acc": acc,
        "Stick": counter["Stick"],
        "Follow": counter["Follow"],
        "Reject": counter["Reject"],
        "N": total,
    }

