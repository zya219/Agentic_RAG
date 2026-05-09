#!/usr/bin/env python3
"""Deterministic prediction evaluator with structured reward components.

Outputs structured semantic rewards and trajectory segments for offline analysis.
These are not token-level PPO rewards yet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter, defaultdict
from typing import Any

ARTICLES = {"a", "an", "the"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold JSONL.")
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--gold_jsonl", required=True)
    parser.add_argument("--output_jsonl", default="results/evaluated_predictions.jsonl")
    parser.add_argument("--summary_json", default="results/evaluation_summary.json")
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(w for w in s.split() if w not in ARTICLES)
    s = " ".join(s.split())
    return s


def em_score(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    ptoks = normalize_answer(pred).split()
    gtoks = normalize_answer(gold).split()
    if not ptoks and not gtoks:
        return 1.0
    if not ptoks or not gtoks:
        return 0.0
    common = Counter(ptoks) & Counter(gtoks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(ptoks)
    r = num_same / len(gtoks)
    return 2 * p * r / (p + r)


def as_gold_list(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(x) for x in answer]
    return [str(answer) if answer is not None else ""]


def extract_trajectory(text: str) -> dict[str, Any]:
    text = text or ""
    valid = [m.strip() for m in re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL) if m.strip()]
    invalid = [m.strip() for m in re.findall(r"<search\s+[^>]*>(.*?)</search>", text, flags=re.DOTALL) if m.strip()]
    ans = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    answer_text = ans[-1].strip() if ans else ""
    spans = {
        "valid_search_spans": [m.span() for m in re.finditer(r"<search>(.*?)</search>", text, flags=re.DOTALL)],
        "answer_spans": [m.span() for m in re.finditer(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)],
    }
    return {
        "valid_search_contents": valid,
        "invalid_search_contents": invalid,
        "answer_text": answer_text,
        "char_spans": spans,
    }


def main() -> None:
    args = parse_args()
    preds = load_jsonl(args.pred_jsonl)
    golds = load_jsonl(args.gold_jsonl)

    gold_by_id = {str(g["id"]): g for g in golds if "id" in g}
    gold_by_q = {g.get("question"): g for g in golds if g.get("question") is not None}

    out_rows = []
    for p in preds:
        g = None
        if "id" in p and str(p["id"]) in gold_by_id:
            g = gold_by_id[str(p["id"])]
        elif p.get("question") in gold_by_q:
            g = gold_by_q[p.get("question")]
        if g is None:
            continue

        raw_text = p.get("output") or p.get("response") or p.get("prediction") or ""
        traj = extract_trajectory(raw_text)
        final_answer = p.get("final_answer") or traj["answer_text"]
        gold_answers = as_gold_list(g.get("answer"))

        best_em = 0.0
        best_f1 = 0.0
        for ga in gold_answers:
            best_em = max(best_em, em_score(final_answer, ga))
            best_f1 = max(best_f1, f1_score(final_answer, ga))

        invalid_search_format = bool(p.get("invalid_search_format", False) or len(traj["invalid_search_contents"]) > 0)
        search_count = int(p.get("search_count", len(traj["valid_search_contents"])))
        expected_search = bool(p.get("expected_search", False))
        actual_search = bool(p.get("actual_search", search_count > 0))
        answer_missing = bool(p.get("answer_missing", False) or not str(final_answer).strip())

        answer_reward = best_f1
        format_reward = -0.1 if invalid_search_format else 0.0
        answer_missing_penalty = -1.0 if answer_missing else 0.0
        efficiency_penalty = -0.05 * max(search_count - 1, 0)
        if expected_search and actual_search:
            search_decision_reward = 0.2
        elif expected_search and not actual_search:
            search_decision_reward = -0.2
        elif (not expected_search) and actual_search:
            search_decision_reward = -0.2
        else:
            search_decision_reward = 0.0
        final_reward = answer_reward + search_decision_reward + format_reward + efficiency_penalty + answer_missing_penalty

        out_rows.append({
            **p,
            "gold_answer": g.get("answer"),
            "answer_em": best_em,
            "answer_f1": best_f1,
            "reward_components": {
                "answer_em": best_em,
                "answer_f1": best_f1,
                "answer_reward": answer_reward,
                "search_decision_reward": search_decision_reward,
                "format_reward": format_reward,
                "efficiency_penalty": efficiency_penalty,
                "answer_missing_penalty": answer_missing_penalty,
                "final_reward": final_reward,
            },
            "trajectory_segments": traj,
        })

    def summarize(rows: list[dict[str, Any]], include_groups: bool = False) -> dict[str, Any]:
        total = len(rows)
        if total == 0:
            return {"total": 0}

        def mean_component(k: str) -> float:
            return sum(r["reward_components"][k] for r in rows) / total

        summary = {
            "total": total,
            "answer_em": sum(r["answer_em"] for r in rows) / total,
            "answer_f1": sum(r["answer_f1"] for r in rows) / total,
            "avg_final_reward": mean_component("final_reward"),
            "avg_answer_reward": mean_component("answer_reward"),
            "avg_search_decision_reward": mean_component("search_decision_reward"),
            "avg_format_reward": mean_component("format_reward"),
            "avg_efficiency_penalty": mean_component("efficiency_penalty"),
            "answer_missing_count": sum(1 for r in rows if r["reward_components"]["answer_missing_penalty"] < 0),
            "search_mismatch_count": sum(1 for r in rows if r["reward_components"]["search_decision_reward"] < 0),
            "invalid_search_format_count": sum(1 for r in rows if r["reward_components"]["format_reward"] < 0),
            "avg_search_count": sum(
                int(r.get("search_count", len(r["trajectory_segments"]["valid_search_contents"])))
                for r in rows
            ) / total,
            "reward_component_means": {
                k: mean_component(k)
                for k in [
                    "answer_em",
                    "answer_f1",
                    "answer_reward",
                    "search_decision_reward",
                    "format_reward",
                    "efficiency_penalty",
                    "answer_missing_penalty",
                    "final_reward",
                ]
            },
        }

        if include_groups:
            for group_key in ["data_source", "split_name"]:
                bucket = defaultdict(list)
                for r in rows:
                    if group_key in r:
                        bucket[str(r[group_key])].append(r)
                if bucket:
                    summary[f"grouped_by_{group_key}"] = {
                        k: summarize(v, include_groups=False)
                        for k, v in bucket.items()
                    }

        return summary

    summary = summarize(out_rows, include_groups=True)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Evaluated rows: {len(out_rows)}")
    print(f"Detailed output: {args.output_jsonl}")
    print(f"Summary output: {args.summary_json}")


if __name__ == "__main__":
    main()
