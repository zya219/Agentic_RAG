#!/usr/bin/env python3
"""Build benchmark/dev/calibration JSONL subsets from parquet datasets.

Policy note:
- Use train/dev splits for debugging, reward calibration, and prompt construction.
- Use test split only for final evaluation; do not tune reward weights on test data.
"""

from __future__ import annotations

import argparse
import json
import os
import math
from typing import Any


QUESTION_CANDIDATES = ["question", "query", "prompt", "input"]
ANSWER_CANDIDATES = ["answer", "answers", "gold_answer", "ground_truth", "target", "golden_answers"]


def detect_field(columns: list[str], candidates: list[str]) -> str:
    lower_to_original = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_to_original:
            return lower_to_original[cand]
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
    try:
        import numpy as np
        np_generic = (np.generic,)
        np_ndarray = (np.ndarray,)
    except Exception:
        np_generic = tuple()
        np_ndarray = tuple()

    """Convert pandas/numpy values into JSON-serializable Python objects."""
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if np_generic and isinstance(value, np_generic):
        return to_jsonable(value.item())

    if np_ndarray and isinstance(value, np_ndarray):
        return [to_jsonable(v) for v in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return to_jsonable(converted)

    try:
        # Handles pandas NA/NaT without ambiguous list/dict/array checks.
        if value != value:
            return None
    except Exception:
        pass

    return str(value)


def main() -> None:
    args = parse_args()
    import pandas as pd

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
