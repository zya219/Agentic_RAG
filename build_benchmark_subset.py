#!/usr/bin/env python3
"""Build benchmark/dev/calibration JSONL subsets from parquet datasets.

Policy note:
- Use train/dev splits for debugging, reward calibration, and prompt construction.
- Use test split only for final evaluation; do not tune reward weights on test data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import pandas as pd


QUESTION_CANDIDATES = ["question", "query", "prompt", "input"]
ANSWER_CANDIDATES = ["answer", "answers", "gold_answer", "golden_answers", "ground_truth", "target"]


def detect_field(columns: list[str], candidates: list[str]) -> str:
    lower_to_original = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    raise ValueError(f"Could not detect required field from candidates={candidates}, columns={columns}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark/dev/calibration JSONL subset from parquet.")
    parser.add_argument("--input_parquet", default="../data/HiPRAG-Dataset/train.parquet")
    parser.add_argument("--output_jsonl", default="results/benchmark_dev_100.jsonl")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_name", default="dev")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    """Convert pandas/numpy/parquet values into JSON-serializable Python objects."""
    try:
        import numpy as np
    except Exception:
        np = None

    if value is None:
        return None

    if np is not None:
        if isinstance(value, np.ndarray):
            return [to_jsonable(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return to_jsonable(value.item())

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    # pandas NA / NaT. Avoid pd.isna on list/dict/array because it can be ambiguous.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass

    return str(value)


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.input_parquet)

    question_field = detect_field(df.columns.tolist(), QUESTION_CANDIDATES)
    answer_field = detect_field(df.columns.tolist(), ANSWER_CANDIDATES)
    data_source_field = "data_source" if "data_source" in df.columns else None
    id_field = "id" if "id" in df.columns else None

    sample_n = min(args.num_samples, len(df))
    sampled = df.sample(n=sample_n, random_state=args.seed) if sample_n < len(df) else df.copy()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    rows = []
    for i, (_, r) in enumerate(sampled.iterrows()):
        row_id = r[id_field] if id_field is not None else i
        rows.append({
            "id": to_jsonable(row_id),
            "question": to_jsonable(r[question_field]),
            "answer": to_jsonable(r[answer_field]),
            "data_source": to_jsonable(r[data_source_field]) if data_source_field is not None else "benchmark_subset",
            "split_name": to_jsonable(args.split_name),
        })

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Detected question field: {question_field}")
    print(f"Detected answer field: {answer_field}")
    print(f"Sampled rows: {len(rows)}")
    print(f"Output JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
