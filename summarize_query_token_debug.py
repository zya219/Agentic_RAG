from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple


DEFAULT_MD = "results/query_token_debug_summary.md"
DEFAULT_CSV = "results/query_token_debug_summary.csv"
DEFAULT_JSON = "results/query_token_debug_summary.json"
DEFAULT_CASES = "results/query_token_case_examples.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize query-token debug JSONL outputs into thesis-ready markdown/csv/json tables."
    )
    p.add_argument("--inputs", nargs="+", required=True, help="MODE=PATH entries")
    p.add_argument("--output_markdown", default=DEFAULT_MD)
    p.add_argument("--output_csv", default=DEFAULT_CSV)
    p.add_argument("--output_json", default=DEFAULT_JSON)
    p.add_argument("--case_markdown", default=DEFAULT_CASES)
    p.add_argument("--max_cases", type=int, default=5)
    return p.parse_args()


def parse_mode_paths(entries: List[str]) -> Tuple[Dict[str, str], List[str]]:
    mode_paths: Dict[str, str] = {}
    warnings: List[str] = []
    for entry in entries:
        if "=" not in entry:
            warnings.append(f"Invalid --inputs entry (missing '='): {entry}")
            continue
        mode, path = entry.split("=", 1)
        mode = mode.strip()
        path = path.strip()
        if not mode or not path:
            warnings.append(f"Invalid --inputs entry (empty mode/path): {entry}")
            continue
        mode_paths[mode] = path
    return mode_paths, warnings


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _safe_mean(total: float, count: int) -> float:
    return float(total / count) if count else 0.0


def summarize_mode(mode: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_rows = len(rows)
    rows_with_search = sum(1 for r in rows if int(r.get("search_action_count", 0)) > 0)
    rows_without_search = total_rows - rows_with_search

    total_search_actions = sum(int(r.get("search_action_count", 0)) for r in rows)
    total_query_text_count = sum(len(r.get("query_texts", []) or []) for r in rows)

    total_action_tokens = 0
    total_query_tokens = 0
    for r in rows:
        action_lists = r.get("action_token_indices", []) or []
        query_lists = r.get("query_token_indices", []) or []
        total_action_tokens += sum(len(x or []) for x in action_lists)
        total_query_tokens += sum(len(x or []) for x in query_lists)

    token_score_sum_total = sum(float(r.get("token_score_sum", 0.0) or 0.0) for r in rows)
    expected_sum_total = sum(float(r.get("expected_reward_sum", 0.0) or 0.0) for r in rows)

    conservation_errors = [float(r.get("reward_conservation_error", 0.0) or 0.0) for r in rows]
    max_conservation_error = max(conservation_errors) if conservation_errors else 0.0
    avg_conservation_error = _safe_mean(sum(conservation_errors), len(conservation_errors))

    warnings_per_row = [r.get("warnings", []) or [] for r in rows]
    warning_count = sum(len(w) for w in warnings_per_row)
    fallback_to_last_token_count = sum(1 for w in warnings_per_row if "fallback_to_last_token" in w)
    fallback_to_coarse_action_count = sum(1 for w in warnings_per_row if "fallback_to_coarse_action" in w)

    return {
        "mode": mode,
        "total_rows": total_rows,
        "rows_with_search": rows_with_search,
        "rows_without_search": rows_without_search,
        "total_search_actions": total_search_actions,
        "avg_search_action_count": _safe_mean(total_search_actions, total_rows),
        "avg_query_text_count": _safe_mean(total_query_text_count, total_rows),
        "avg_action_token_count_per_action": _safe_mean(total_action_tokens, total_search_actions),
        "avg_query_token_count_per_action": _safe_mean(total_query_tokens, total_search_actions),
        "avg_token_score_sum": _safe_mean(token_score_sum_total, total_rows),
        "avg_expected_reward_sum": _safe_mean(expected_sum_total, total_rows),
        "max_reward_conservation_error": max_conservation_error,
        "avg_reward_conservation_error": avg_conservation_error,
        "warning_count": warning_count,
        "fallback_to_last_token_count": fallback_to_last_token_count,
        "fallback_to_coarse_action_count": fallback_to_coarse_action_count,
    }


def write_markdown(path: str, summaries: List[Dict[str, Any]]) -> None:
    headers = [
        "Mode",
        "Rows",
        "Rows w/ Search",
        "Avg Search Actions",
        "Avg Action Tokens",
        "Avg Query Tokens",
        "Avg Score Sum",
        "Avg Expected Sum",
        "Max Conservation Error",
        "Warnings",
    ]
    lines = [
        "# Query-Token Debug Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for s in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    s["mode"],
                    str(s["total_rows"]),
                    str(s["rows_with_search"]),
                    f"{s['avg_search_action_count']:.6f}",
                    f"{s['avg_action_token_count_per_action']:.6f}",
                    f"{s['avg_query_token_count_per_action']:.6f}",
                    f"{s['avg_token_score_sum']:.6f}",
                    f"{s['avg_expected_reward_sum']:.6f}",
                    f"{s['max_reward_conservation_error']:.6f}",
                    str(s["warning_count"]),
                ]
            )
            + " |"
        )
    write_text(path, "\n".join(lines) + "\n")


def write_csv(path: str, summaries: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "Mode",
        "Rows",
        "Rows w/ Search",
        "Avg Search Actions",
        "Avg Action Tokens",
        "Avg Query Tokens",
        "Avg Score Sum",
        "Avg Expected Sum",
        "Max Conservation Error",
        "Warnings",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow(
                {
                    "Mode": s["mode"],
                    "Rows": s["total_rows"],
                    "Rows w/ Search": s["rows_with_search"],
                    "Avg Search Actions": f"{s['avg_search_action_count']:.6f}",
                    "Avg Action Tokens": f"{s['avg_action_token_count_per_action']:.6f}",
                    "Avg Query Tokens": f"{s['avg_query_token_count_per_action']:.6f}",
                    "Avg Score Sum": f"{s['avg_token_score_sum']:.6f}",
                    "Avg Expected Sum": f"{s['avg_expected_reward_sum']:.6f}",
                    "Max Conservation Error": f"{s['max_reward_conservation_error']:.6f}",
                    "Warnings": s["warning_count"],
                }
            )


def write_json(path: str, summaries: List[Dict[str, Any]], notes: List[str]) -> None:
    payload = {
        "summary_version": "v1",
        "metrics": summaries,
        "notes": notes,
    }
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_case_markdown(path: str, rows: List[Dict[str, Any]], max_cases: int) -> None:
    selected = [r for r in rows if int(r.get("search_action_count", 0)) > 0][: max(0, max_cases)]
    lines = ["# Query-Token Case Examples", ""]
    if not selected:
        lines.extend(["No eligible cases found.", ""])
    for i, r in enumerate(selected, start=1):
        lines.extend(
            [
                f"## Case {i}: {r.get('id', '')}",
                "",
                f"- **id**: `{r.get('id', '')}`",
                f"- **question**: {r.get('question', '')}",
                f"- **query_texts**: `{json.dumps(r.get('query_texts', []), ensure_ascii=False)}`",
                f"- **action_char_spans**: `{json.dumps(r.get('action_char_spans', []), ensure_ascii=False)}`",
                f"- **query_char_spans**: `{json.dumps(r.get('query_char_spans', []), ensure_ascii=False)}`",
                f"- **action_token_indices**: `{json.dumps(r.get('action_token_indices', []), ensure_ascii=False)}`",
                f"- **query_token_indices**: `{json.dumps(r.get('query_token_indices', []), ensure_ascii=False)}`",
                f"- **token_score_sum**: {float(r.get('token_score_sum', 0.0) or 0.0):.6f}",
                f"- **expected_reward_sum**: {float(r.get('expected_reward_sum', 0.0) or 0.0):.6f}",
                f"- **reward_conservation_error**: {float(r.get('reward_conservation_error', 0.0) or 0.0):.6f}",
                f"- **warnings**: `{json.dumps(r.get('warnings', []), ensure_ascii=False)}`",
                f"- **result_preview**: {r.get('result_preview', '')}",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    args = parse_args()
    mode_paths, parse_warnings = parse_mode_paths(args.inputs)
    notes: List[str] = list(parse_warnings)

    mode_rows: Dict[str, List[Dict[str, Any]]] = {}
    for mode, path in mode_paths.items():
        if not os.path.exists(path):
            warning = f"Missing input file for mode '{mode}': {path}"
            print(f"[warning] {warning}", file=sys.stderr)
            notes.append(warning)
            continue
        try:
            rows = read_jsonl(path)
        except ValueError as exc:
            print(f"[warning] {exc}", file=sys.stderr)
            notes.append(str(exc))
            continue
        mode_rows[mode] = rows

    if not mode_rows:
        print("[error] No valid input JSONL files were found.", file=sys.stderr)
        return 1

    summaries = [summarize_mode(mode, rows) for mode, rows in mode_rows.items()]
    write_markdown(args.output_markdown, summaries)
    write_csv(args.output_csv, summaries)
    write_json(args.output_json, summaries, notes)

    case_source_rows = mode_rows.get("query_token_uniform", [])
    write_case_markdown(args.case_markdown, case_source_rows, args.max_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
