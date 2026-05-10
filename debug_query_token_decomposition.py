from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from search_r1.query_token_decomposition import allocate_reward_to_tokens, build_query_token_masks


def parse_args():
    p = argparse.ArgumentParser(description="Debug post-hoc query-token-level action decomposition.")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--text_field", default="result")
    p.add_argument("--mode", choices=["none", "coarse_action", "query_token_uniform"], default="none")
    p.add_argument("--search_reward_value", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    outputs: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = str(row.get(args.text_field, ""))
        masks = build_query_token_masks(text, tokenizer)
        response_len = len(masks["tokens"])

        # Mode semantics:
        # - none: no-allocation baseline (all zero token scores)
        # - coarse_action: allocate to the whole <search>...</search> action span
        # - query_token_uniform: allocate to query tokens only, fallback to coarse action then last token
        warnings = []
        if args.mode == "none":
            scores = [0.0] * response_len
            warn = None
        else:
            selected: List[int] = []
            if args.mode == "coarse_action":
                selected = [idx for idx, v in enumerate(masks["action_mask"]) if v > 0]
            elif args.mode == "query_token_uniform":
                selected = [idx for idx, v in enumerate(masks["query_token_mask"]) if v > 0]
                if not selected:
                    fallback = [idx for idx, v in enumerate(masks["action_mask"]) if v > 0]
                    if fallback:
                        selected = fallback
                        warnings.append("fallback_to_coarse_action")
                    elif response_len > 0:
                        selected = [response_len - 1]
                        warnings.append("fallback_to_last_token")
            scores, warn = allocate_reward_to_tokens(response_len, selected, args.search_reward_value)

        if warn:
            warnings.append(warn)

        score_sum = float(sum(scores))
        expected = 0.0 if args.mode == "none" else float(args.search_reward_value)
        outputs.append(
            {
                "id": row.get("id", i),
                "question": row.get("question", ""),
                "mode": args.mode,
                "result_preview": text[:200],
                "search_action_count": len(masks["search_actions"]),
                "query_texts": [a["query_text"] for a in masks["search_actions"]],
                "action_char_spans": [a["action_char_span"] for a in masks["search_actions"]],
                "query_char_spans": [a["query_char_span"] for a in masks["search_actions"]],
                "action_token_indices": [a["action_token_indices"] for a in masks["search_actions"]],
                "query_token_indices": [a["query_token_indices"] for a in masks["search_actions"]],
                "token_score_sum": score_sum,
                "expected_reward_sum": expected,
                "reward_conservation_error": abs(score_sum - expected),
                "warnings": warnings,
            }
        )

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
