from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from tiebe_rag_conflict.eval.metrics import compute_csi
from tiebe_rag_conflict.eval.metrics import classify
from tiebe_rag_conflict.utils.io import read_jsonl

LOGGER = logging.getLogger(__name__)


def _year_bucket(year: int | None, month: int | None, months_per_bucket: int) -> str:
    if year is None or month is None:
        return "unknown"
    if months_per_bucket >= 12:
        return str(year)
    month = max(1, min(12, month))
    bucket_idx = (month - 1) // months_per_bucket
    start_month = bucket_idx * months_per_bucket + 1
    end_month = min(start_month + months_per_bucket - 1, 12)
    return f"{year}-{start_month:02d}~{year}-{end_month:02d}"


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("preds_*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def build_slice_report(rows: list[dict[str, Any]], months_per_bucket: int) -> pd.DataFrame:
    if not rows:
        raise ValueError("No prediction rows found. Run evaluation first.")
    df = pd.DataFrame(rows)
    df["label"] = [
        classify(row.get("pred", ""), row.get("gold", ""), row.get("false")) for row in rows
    ]
    df["year_bucket"] = [
        _year_bucket(row.get("year"), row.get("month"), months_per_bucket) for row in rows
    ]
    group_cols = ["mode", "ctx_type", "region", "language", "year_bucket"]
    agg_rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        group_dicts = group[["pred", "gold", "false"]].to_dict(orient="records")
        metrics = compute_csi(group_dicts)
        row_dict = {col: key for col, key in zip(group_cols, keys)}
        row_dict.update(metrics)
        agg_rows.append(row_dict)
    result = pd.DataFrame(agg_rows)
    result.sort_values(group_cols, inplace=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slice report for TiEBe RAG evaluation")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--bucket", type=int, default=6, help="Months per year bucket")
    parser.add_argument("--out", default=None, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = _load_rows(run_dir)
    report = build_slice_report(rows, args.bucket)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = run_dir / f"slice_report_{args.bucket}m.csv"
    report.to_csv(out_path, index=False)
    LOGGER.info("Saved slice report to %s", out_path)


if __name__ == "__main__":  # pragma: no cover
    main()

