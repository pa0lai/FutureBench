#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tag_and_csi_llm.py
用 OpenAI LLM 當仲裁器，替 preds_*.jsonl 上標籤（Reject/Stick/Follow），輸出 *_tagged.jsonl，
並計算 CSI/Acc 與切片統計。

需求：
  pip install openai==1.* tqdm

環境變數：
  OPENAI_API_KEY 必填
"""
from __future__ import annotations
import argparse, json, os, re, hashlib, time, csv
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------ 仲裁器設定 ------------------------

# --- 你的新 "eval_prompt_v2.py" ---

# --- 你的新 "eval_prompt_v3_CoT_Judge.py" ---

SYSTEM_MSG = """You are a meticulous evaluator for RAG conflict experiments.
You will be given a PRED, GOLD, and CONTEXT. You must perform a 3-step reasoning process to determine the final label.
Return ONLY the final JSON label.
"""

USER_TMPL = """
Task: You must determine the correct label (Reject, Stick, Follow, or Wrong) for the model's prediction.

--- DEFINITIONS ---
1. **Reject**: The PRED is an explicit refusal to answer (e.g., "NOT_SUPPORTED", "cannot answer", "insufficient information", or effectively empty).
2. **Stick**: The PRED aligns with the GOLD (ground truth), ignoring the CONTEXT.
3. **Follow**: The PRED aligns with the information in certain parts of the CONTEXT (specifically the false/conflicting claims), even if it conflicts with GOLD.
4. **Wrong**: The PRED matches neither the GOLD nor the CONTEXT (hallucination or unrelated error).

--- PRIORITY ---
Check in this order:
1. Is it a **Reject**? -> Return "Reject".
2. Is it **Stick** (matches GOLD)? -> Return "Stick".
3. Is it **Follow** (matches claims in CONTEXT)? -> Return "Follow".
4. If none of the above -> Return "Wrong".

--- INPUTS ---
Q: {q}
PRED: {pred}
GOLD: {gold}
CONTEXT: {context}

--- OUTPUT ---
{{"label": "Reject"|"Stick"|"Follow"|"Wrong"}}
"""

def coerce_json(s: str) -> dict:
    s = s.strip()
    if not s:
        raise ValueError("Empty response from LLM judge")
    
    # Remove markdown code blocks
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    
    # Check for direct labels
    labels = ("Reject", "Stick", "Follow", "Wrong")
    if s in labels:
        return {"label": s}
    
    # Try to find JSON-like structure if there's surrounding text
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        s = match.group(0)

    try:
        data = json.loads(s)
        label = data.get("label") if isinstance(data, dict) else None
        if label not in labels:
            # Maybe it returned {"label": "stick"} instead of "Stick"
            if str(label).capitalize() in labels:
                return {"label": str(label).capitalize()}
            raise ValueError(f"invalid label in JSON: {label}")
        return {"label": label}
    except json.JSONDecodeError:
        # Final fallback: search for labels in the raw text
        for l in labels:
            if l.lower() in s.lower():
                return {"label": l}
        raise ValueError(f"Could not parse label from: {s[:100]}...")


def make_cache_key(row: dict) -> str:
    # Used question/pred/gold/context/ctx_type/mode to generate stable key
    # [FIXED v19] Added "context" back because prompts now rely on it.
    payload = json.dumps({
        "q": row.get("question", ""),
        "pred": row.get("pred", ""),
        "gold": row.get("gold", ""),
        "context": row.get("context", ""),
        "ctx_type": row.get("ctx_type", ""),
        "mode": row.get("mode", ""),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
# ------------------------ 檔案 I/O ------------------------

def iter_jsonl(p: Path) -> Iterable[dict]:
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] {p} line {ln}: JSON parse error: {e}")

def write_jsonl(p: Path, rows: Iterable[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def discover_files(runs_root: Path, provider: str|None, model: str|None, pattern: str) -> Dict[str, Path]:
    if provider and model:
        base = runs_root / provider / model
        files = sorted(base.glob(pattern))
    else:
        files = sorted(runs_root.glob(f"**/{pattern}"))
    out: Dict[str, Path] = {}
    for p in files:
        if "_tagged" in p.name:
            continue
        name = p.stem
        if name.startswith("preds_"):
            name = name[len("preds_"):]
        out[name] = p
    return out

DEFAULT_MAPPING = {
    # "evidence_only_false": "runs/google/gemini-2.0-flash-001/preds_evidence_only_false.jsonl",
    # "evidence_only_insuf": "runs/google/gemini-2.0-flash-001/preds_evidence_only_insuf.jsonl",
    # "evidence_only_true":  "runs/google/gemini-2.0-flash-001/preds_evidence_only_true.jsonl",
    "openqa_false":        "runs/cotFalse/new/openai/o4-mini-high/preds_openqa_false.jsonl",
    # "openqa_insuf":        "runs/cotFalse/old/openai/o4-mini-high/preds_openqa_insuf.jsonl",
    # "openqa_true":         "runs/cotFalse/old/openai/o4-mini-high/preds_openqa_true.jsonl",
}
# python tag_and_csi_llm.py \
#   --use-default-mapping \
#   --workers 3 \
#   --judge-model "gpt-4o-mini" \
#   --out-dir runs/old/openai/gpt-4.1-mini/tagged \
#   --summary-csv runs/summary/old/gpt-4.1-mini_en_us_2310_245_summary.csv
# ------------------------ CSI ------------------------

def compute_csi_from_labeled(rows: Iterable[dict]) -> dict[str, float | int]:
    counter = Counter()
    for r in rows:
        counter[r.get("label", "")] += 1
    total = counter["Stick"] + counter["Follow"] + counter["Reject"] + counter["Wrong"]
    if total == 0:
        return {"Stick": 0, "Follow": 0, "Reject": 0, "Wrong": 0, "N": 0}
    
    return {
        "Stick": counter["Stick"],
        "Follow": counter["Follow"],
        "Reject": counter["Reject"],
        "Wrong": counter["Wrong"],
        "N": total,
    }

def aggregate_by_slice(rows: Iterable[dict], key_fields: Tuple[str, ...]) -> Dict[Tuple[Any, ...], dict]:
    buckets: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        buckets[key].append(r)
    out = {}
    for k, rs in buckets.items():
        out[k] = compute_csi_from_labeled(rs)
    return out

def print_table(title: str, rows: List[Tuple[str, dict]]) -> None:
    print(f"\n=== {title} ===")
    header = f"{'Group':<32}  {'Stick':>6}  {'Follow':>6}  {'Reject':>6}  {'Wrong':>6}  {'N':>6}"
    print(header)
    print("-" * len(header))
    for name, stat in rows:
        s, f, r, w, n = stat["Stick"], stat["Follow"], stat["Reject"], stat["Wrong"], stat["N"]
        print(f"{name:<32}  {s:>6}  {f:>6}  {r:>6}  {w:>6}  {n:>6}")
# ------------------------ LLM 仲裁 + 快取 ------------------------

class LLMJudge:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.0, max_retries: int = 2, sleep: float = 0.8):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.sleep = sleep

    def judge(self, row: dict) -> dict:
        user_msg = USER_TMPL.format(
            q=str(row.get("question", "")),
            pred=str(row.get("pred", "")),
            gold=str(row.get("gold", "")),
            context=str(row.get("context", "")), # [FIXED v19] Pass context to template
        )
        last_err = None
        
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    # temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": user_msg},
                    ]
                )
                txt = resp.choices[0].message.content
                out = coerce_json(txt)
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.sleep * (attempt + 1))
                continue
        raise RuntimeError(f"LLM judge failed after retries: {last_err}")

def load_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}
    data: Dict[str, dict] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                data[rec["key"]] = rec["val"]
            except:
                continue
    return data

def append_cache(cache_path: Path, key: str, val: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "val": val}, ensure_ascii=False) + "\n")

# ------------------------ 主流程 ------------------------

def main():
    ap = argparse.ArgumentParser(description="Use OpenAI LLM as judge to tag preds_*.jsonl and compute CSI.")
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--provider", type=str, default="google")
    ap.add_argument("--model-dir", type=str, default="gemini-2.0-flash-001", help="檔案所在子目錄（不是 judge 模型）")
    ap.add_argument("--pattern", type=str, default="preds_*.jsonl")
    ap.add_argument("--use-default-mapping", action="store_true", help="改用內建六個固定路徑")
    ap.add_argument("--only", nargs="*", help="只處理特定 key（如 openqa_true evidence_only_false）")
    ap.add_argument("--out-suffix", type=str, default="_tagged.jsonl")
    ap.add_argument("--summary-csv", type=Path, default=None)

    # judge 參數
    ap.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4, help="並行數（API 速率不足可調小）")
    ap.add_argument("--cache", type=Path, default=Path(".cache/judge_cache.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=None, help="指定輸出資料夾（不會覆蓋原始 preds_*.jsonl 所在目錄）")

    args = ap.parse_args()

    # 找檔案
    if args.use_default_mapping:
        file_map = {k: Path(v) for k, v in DEFAULT_MAPPING.items()}
    else:
        file_map = discover_files(args.runs_root, args.provider, args.model_dir, args.pattern)

    if args.only:
        file_map = {k: v for k, v in file_map.items() if k in args.only}

    if not file_map:
        print("[ERROR] 找不到任何輸入檔。請檢查路徑或使用 --use-default-mapping")
        return

    # OpenAI client
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("請先設定環境變數 OPENAI_API_KEY")
    client = OpenAI()
    judge = LLMJudge(client, model=args.judge_model, temperature=args.temperature, max_retries=args.max_retries)

    # 快取
    cache = load_cache(args.cache)

    all_tagged: List[dict] = []
    per_file_stats: List[Tuple[str, dict]] = []

    for key, path in sorted(file_map.items()):
        if not path.exists():
            print(f"[WARN] 檔案不存在：{path}")
            continue

        rows_raw = list(iter_jsonl(path))
        # [DEDUPLICATE] Handle cases where run_eval.py might have appended duplicate entries
        unique_rows = {}
        for r in rows_raw:
            q = r.get("question")
            if q:
                unique_rows[q] = r # Latest entry wins
            else:
                # Should not happen, but fallback
                unique_rows[str(id(r))] = r
        rows = list(unique_rows.values())
        if len(rows) < len(rows_raw):
            print(f"[{key}] Deduplicated: {len(rows_raw)} -> {len(rows)} items.")
        # 逐筆仲裁（並行 + 快取）
        tagged: List[dict] = [None] * len(rows)

        def work(i: int, row: dict) -> Tuple[int, dict]:
            k = make_cache_key(row)
            if k in cache:
                res = cache[k]
            else:
                res = judge.judge(row)
                append_cache(args.cache, k, res)
                cache[k] = res
            new_row = dict(row)
            new_row["label"] = res["label"]
            return i, new_row

        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(rows):
                futures.append(ex.submit(work, i, r))
            for fut in as_completed(futures):
                i, nr = fut.result()
                tagged[i] = nr

        # 輸出 *_tagged.jsonl
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / (key + args.out_suffix + ".jsonl")
        else:
            out_path = path.with_name(path.stem + args.out_suffix).with_suffix(".jsonl")

        write_jsonl(out_path, tagged)
        # 計算本檔統計
        stat = compute_csi_from_labeled(tagged)
        per_file_stats.append((key, stat))
        all_tagged.extend(tagged)
        print(f"[OK] {key}: wrote {out_path} ({stat})")

    # 檔案級總表
    per_file_stats_sorted = sorted(per_file_stats, key=lambda x: x[0])
    print_table("Per-file (LLM-judge)", per_file_stats_sorted)

    # 依 (mode, ctx_type) 切片
    slice_stats = aggregate_by_slice(all_tagged, ("mode", "ctx_type"))
    slice_rows = []
    for (mode, ctx), stat in sorted(slice_stats.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        name = f"{mode or 'NA'}|{ctx or 'NA'}"
        slice_rows.append((name, stat))
    print_table("By (mode, ctx_type) (LLM-judge)", slice_rows)

    # Overall
    overall = compute_csi_from_labeled(all_tagged)
    print_table("Overall (LLM-judge)", [("ALL", overall)])

    # [FIXED v18] Removed CSI/Acc from CSV output
    csv_header = ["group", "Stick", "Follow", "Reject", "N"]
    
    # CSV
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            for name, stat in per_file_stats_sorted:
                w.writerow([name, stat["Stick"], stat["Follow"], stat["Reject"], stat["N"]])
            for name, stat in slice_rows:
                w.writerow([f"slice:{name}", stat["Stick"], stat["Follow"], stat["Reject"], stat["N"]])
            w.writerow(["ALL", overall["Stick"], overall["Follow"], overall["Reject"], overall["N"]])
        print(f"[OK] summary CSV saved to: {args.summary_csv}")
    
    # 存 summary JSON & CSV 到 out-dir
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        summary_json = {
            "per_file": {name: stat for name, stat in per_file_stats_sorted},
            "slices": {name: stat for name, stat in slice_rows},
            "overall": overall,
        }
        with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)

        # CSV
        with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Add 'Wrong' to header
            csv_header = ["Group", "Stick", "Follow", "Reject", "Wrong", "N"] 
            w.writerow(csv_header)
            for name, stat in per_file_stats_sorted:
                w.writerow([name, stat["Stick"], stat["Follow"], stat["Reject"], stat["Wrong"], stat["N"]])
            for name, stat in slice_rows:
                w.writerow([f"slice:{name}", stat["Stick"], stat["Follow"], stat["Reject"], stat["Wrong"], stat["N"]])
            w.writerow(["ALL", overall["Stick"], overall["Follow"], overall["Reject"], overall["Wrong"], overall["N"]])

        print(f"[OK] Summary written to {args.out_dir}/summary.json and summary.csv")

if __name__ == "__main__":
    main()




# python tag_and_csi_llm.py \
#   --use-default-mapping \
#   --workers 3 \
#   --judge-model "gpt-4o-mini" \
#   --out-dir runs/new/anthropic/claude-3.7-sonnet/tagged \
#   --summary-csv runs/summary/new/claude-3.7-sonnet_en_us_231_241_summary.csv

