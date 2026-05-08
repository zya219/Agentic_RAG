import argparse
import json
from collections import Counter




def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_bool(value):
    return bool(value)


def summarize(rows):
    total = len(rows)

    manual_correct = sum(_to_bool(r.get("manual_correct")) for r in rows)
    needs_retrieval_gold_true = sum(_to_bool(r.get("needs_retrieval_gold")) for r in rows)
    expected_search_true = sum(_to_bool(r.get("expected_search")) for r in rows)
    actual_search_true = sum(_to_bool(r.get("actual_search")) for r in rows)
    search_mismatch_count = sum(_to_bool(r.get("search_mismatch")) for r in rows)
    answer_missing_count = sum(_to_bool(r.get("answer_missing")) for r in rows)

    routing_gold_match = sum(
        _to_bool(r.get("expected_search")) == _to_bool(r.get("needs_retrieval_gold"))
        for r in rows
    )
    actual_search_gold_match = sum(
        _to_bool(r.get("actual_search")) == _to_bool(r.get("needs_retrieval_gold"))
        for r in rows
    )

    error_type_counts = Counter(str(r.get("error_type", "")).strip() or "unknown" for r in rows)

    return {
        "total": total,
        "manual_correct": manual_correct,
        "manual_wrong": total - manual_correct,
        "manual_accuracy": (manual_correct / total) if total else 0.0,
        "needs_retrieval_gold_true": needs_retrieval_gold_true,
        "needs_retrieval_gold_false": total - needs_retrieval_gold_true,
        "expected_search_true": expected_search_true,
        "actual_search_true": actual_search_true,
        "routing_gold_accuracy": (routing_gold_match / total) if total else 0.0,
        "actual_search_gold_accuracy": (actual_search_gold_match / total) if total else 0.0,
        "search_mismatch_count": search_mismatch_count,
        "answer_missing_count": answer_missing_count,
        "error_type_counts": dict(error_type_counts),
    }


def is_problem_case(row):
    if not _to_bool(row.get("manual_correct")):
        return True
    error_type = str(row.get("error_type", "")).strip().lower()
    return error_type != "correct"


def main():
    parser = argparse.ArgumentParser(description="Analyze manually annotated inference JSONL results.")
    parser.add_argument("--input_jsonl", default="results/manual_eval_10case.jsonl")
    parser.add_argument("--summary_json", default="results/manual_eval_summary.json")
    parser.add_argument("--problem_jsonl", default="results/manual_eval_problem_cases.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    summary = summarize(rows)
    problem_cases = [row for row in rows if is_problem_case(row)]

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_jsonl(problem_cases, args.problem_jsonl)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary to: {args.summary_json}")
    print(f"Wrote problem cases to: {args.problem_jsonl}")


if __name__ == "__main__":
    main()
