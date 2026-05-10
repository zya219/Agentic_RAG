#!/usr/bin/env python3
"""Check experiment output files exist and have basic expected structure."""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_FIELDS = [
    "id",
    "question",
    "result",
    "final_answer",
    "has_search",
    "search_queries",
    "search_count",
    "decision_type",
    "answer_missing",
]

SUMMARY_FIELDS = [
    "total",
    "answer_em",
    "answer_f1",
    "avg_search_count",
    "avg_final_reward",
    "answer_missing_count",
    "invalid_search_format_count",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment output artifacts.")
    parser.add_argument("--prediction_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--expected_rows", type=int, default=None)
    parser.add_argument("--require_fields", nargs="+", default=DEFAULT_FIELDS)
    args = parser.parse_args()

    failed = False

    def check(condition: bool, message: str) -> None:
        nonlocal failed
        if condition:
            print(f"PASS: {message}")
        else:
            print(f"FAIL: {message}")
            failed = True

    check(os.path.exists(args.prediction_jsonl), f"prediction_jsonl exists: {args.prediction_jsonl}")
    check(os.path.exists(args.summary_json), f"summary_json exists: {args.summary_json}")

    checked_rows = []
    line_count = 0
    if os.path.exists(args.prediction_jsonl):
        with open(args.prediction_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
                    if len(checked_rows) < 3:
                        checked_rows.append(line)

    if args.expected_rows is not None:
        check(line_count == args.expected_rows, f"prediction_jsonl row count == {args.expected_rows} (actual: {line_count})")

    parsed_rows = []
    for idx, line in enumerate(checked_rows, start=1):
        try:
            parsed_rows.append(json.loads(line))
            check(True, f"prediction_jsonl row {idx} valid JSON")
        except json.JSONDecodeError as exc:
            check(False, f"prediction_jsonl row {idx} valid JSON ({exc})")

    for idx, row in enumerate(parsed_rows, start=1):
        missing = [field for field in args.require_fields if field not in row]
        check(not missing, f"prediction_jsonl row {idx} has required fields (missing: {missing})")

    if os.path.exists(args.summary_json):
        try:
            with open(args.summary_json, "r", encoding="utf-8") as f:
                summary = json.load(f)
            check(True, "summary_json is valid JSON")
            missing = [field for field in SUMMARY_FIELDS if field not in summary]
            check(not missing, f"summary_json has required fields (missing: {missing})")
        except (OSError, json.JSONDecodeError) as exc:
            check(False, f"summary_json is valid JSON ({exc})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
