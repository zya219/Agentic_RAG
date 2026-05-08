import argparse
import json
from collections import defaultdict

from utils import load_jsonl, write_jsonl


def _to_bool(v):
    return bool(v)


def summarize(rows):
    total = len(rows)
    decision_true = sum(_to_bool(r.get("decision")) for r in rows)
    actual_search = sum(_to_bool(r.get("has_search", r.get("actual_search"))) for r in rows)
    invalid_search_format_count = sum(_to_bool(r.get("invalid_search_format")) for r in rows)
    valid_search_count = sum((r.get("search_action_status") == "valid_search") for r in rows)
    no_search_action_count = sum((r.get("search_action_status") == "no_search") for r in rows)
    search_action_status_counts = {
        "valid_search": valid_search_count,
        "invalid_search_format": sum((r.get("search_action_status") == "invalid_search_format") for r in rows),
        "no_search": no_search_action_count,
    }
    mismatch = sum(_to_bool(r.get("search_mismatch")) or (_to_bool(r.get("decision")) and not _to_bool(r.get("has_search", r.get("actual_search")))) for r in rows)
    overuse = sum(_to_bool(r.get("search_overuse")) or ((not _to_bool(r.get("decision"))) and _to_bool(r.get("has_search", r.get("actual_search")))) for r in rows)
    search_counts = [int(r.get("search_count", 0) or 0) for r in rows]
    ans_missing = sum(_to_bool(r.get("answer_missing")) or not str(r.get("final_answer") or "").strip() for r in rows)
    return {
        "total_sample_count": total,
        "decision_true_count": decision_true,
        "decision_false_count": total - decision_true,
        "actual_search_count": actual_search,
        "no_search_count": total - actual_search,
        "valid_search_count": valid_search_count,
        "no_search_action_count": no_search_action_count,
        "invalid_search_format_count": invalid_search_format_count,
        "invalid_search_format_rate": (invalid_search_format_count / total) if total else 0.0,
        "search_action_status_counts": search_action_status_counts,
        "search_mismatch_count": mismatch,
        "search_mismatch_rate": (mismatch / total) if total else 0.0,
        "search_overuse_count": overuse,
        "avg_search_count": (sum(search_counts) / total) if total else 0.0,
        "answer_missing_count": ans_missing,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze decision vs actual search behavior from inference JSONL outputs.")
    parser.add_argument("--input_jsonl", default="results/hf_test_output.jsonl")
    parser.add_argument("--summary_json", default="results/hf_test_summary.json")
    parser.add_argument("--mismatch_jsonl", default="results/search_mismatch_cases.jsonl")
    parser.add_argument("--invalid_search_jsonl", default="results/invalid_search_format_cases.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    overall = summarize(rows)

    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("decision_type", "unknown")].append(r)

    grouped = {k: summarize(v) for k, v in by_type.items()}

    mismatch_rows = [
        r for r in rows
        if (_to_bool(r.get("search_mismatch")) or (_to_bool(r.get("decision")) and not _to_bool(r.get("has_search", r.get("actual_search")))))
    ]
    invalid_search_rows = [r for r in rows if _to_bool(r.get("invalid_search_format"))]

    summary = {"overall": overall, "by_decision_type": grouped}
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_jsonl(mismatch_rows, args.mismatch_jsonl)
    write_jsonl(invalid_search_rows, args.invalid_search_jsonl)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary to: {args.summary_json}")
    print(f"Wrote mismatch cases to: {args.mismatch_jsonl}")
    print(f"Wrote invalid search format cases to: {args.invalid_search_jsonl}")


if __name__ == "__main__":
    main()
