#!/usr/bin/env python3
"""Aggregate baseline summary JSON files into markdown/csv/json reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

DEFAULT_WARNING = (
    "avg_final_reward may not be directly comparable across baselines if "
    "search_decision_reward gives fixed reward to forced-search methods. "
    "Answer EM/F1 and avg_search_count should be interpreted together."
)

METRICS = [
    "total",
    "answer_em",
    "answer_f1",
    "avg_search_count",
    "search_mismatch_count",
    "invalid_search_format_count",
    "answer_missing_count",
    "avg_final_reward",
    "avg_answer_reward",
    "avg_search_decision_reward",
    "avg_format_reward",
    "avg_efficiency_penalty",
]

TABLE_COLUMNS = [
    ("method", "Method"),
    ("total", "Total"),
    ("answer_em", "EM"),
    ("answer_f1", "F1"),
    ("avg_search_count", "Avg Search"),
    ("search_mismatch_count", "Search Mismatch"),
    ("invalid_search_format_count", "Invalid Search Format"),
    ("answer_missing_count", "Answer Missing"),
    ("avg_answer_reward", "Avg Answer Reward"),
    ("avg_final_reward", "Avg Final Reward"),
]


def parse_inputs(entries: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --inputs entry '{entry}', expected METHOD=PATH")
        method, path = entry.split("=", 1)
        method, path = method.strip(), path.strip()
        if not method or not path:
            raise ValueError(f"Invalid --inputs entry '{entry}', empty method or path")
        parsed.append((method, path))
    return parsed


def to_display(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value != 0 else "0"
    return str(value)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate baseline summary JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="METHOD=PATH_TO_SUMMARY_JSON entries")
    parser.add_argument("--output_markdown", default="results/baseline_comparison.md")
    parser.add_argument("--output_csv", default="results/baseline_comparison.csv")
    parser.add_argument("--output_json", default="results/baseline_comparison.json")
    parser.add_argument("--title", default="Baseline Comparison")
    args = parser.parse_args()

    entries = parse_inputs(args.inputs)

    methods: List[Dict[str, object]] = []
    warnings: List[str] = []

    for method, path in entries:
        if not os.path.exists(path):
            msg = f"[WARN] Missing summary file for method '{method}': {path}"
            print(msg, file=sys.stderr)
            warnings.append(msg)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"[WARN] Failed to read summary for method '{method}' ({path}): {exc}"
            print(msg, file=sys.stderr)
            warnings.append(msg)
            continue

        row: Dict[str, object] = {"method": method}
        for metric in METRICS:
            row[metric] = data.get(metric)
        methods.append(row)

    if not methods:
        print("[ERROR] No valid summary file found from --inputs entries.", file=sys.stderr)
        return 1

    ensure_parent(args.output_markdown)
    ensure_parent(args.output_csv)
    ensure_parent(args.output_json)

    with open(args.output_markdown, "w", encoding="utf-8") as f:
        f.write(f"# {args.title}\n\n")
        f.write(f"> Warning: {DEFAULT_WARNING}\n\n")
        f.write("| " + " | ".join(label for _, label in TABLE_COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |\n")
        for row in methods:
            f.write("| " + " | ".join(to_display(row.get(key)) for key, _ in TABLE_COLUMNS) + " |\n")

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in TABLE_COLUMNS])
        for row in methods:
            writer.writerow([row.get(key) for key, _ in TABLE_COLUMNS])

    normalized = {
        "title": args.title,
        "methods": methods,
        "metrics": METRICS,
        "notes": [DEFAULT_WARNING] + warnings,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote markdown: {args.output_markdown}")
    print(f"[OK] Wrote csv: {args.output_csv}")
    print(f"[OK] Wrote json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
