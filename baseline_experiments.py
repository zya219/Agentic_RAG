#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any
import json



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline/ablation inference experiments.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tokenizer_id", default=None)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--mode", required=True, choices=["direct", "always_retrieve", "rule_based", "prompt_self_routing"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from inference import (
        AGENT_PROMPT_V2_SHORT,
        DIRECT_PROMPT,
        build_diagnostic_fields,
        decide_retrieval,
        extract_final_answer,
        extract_invalid_search_queries,
        extract_search_queries,
        inference_hf_single,
        inference_hf_single_direct,
        search,
    )

    def format_context(search_result: str) -> str:
        return f"<context>{search_result}</context>"

    def call_retriever(query: str, retriever_url: str) -> str:
        return search(query, retriever_url=retriever_url)

    def detect_invalid_search_format(result: str) -> tuple[bool, list[str]]:
        invalid = extract_invalid_search_queries(result)
        return len(invalid) > 0, invalid

    def build_output_row(src: dict[str, Any], result: str, decision: bool, decision_type: str, expected_search: bool | None = None) -> dict[str, Any]:
        search_queries = extract_search_queries(result)
        has_search = len(search_queries) > 0
        final_answer = extract_final_answer(result)
        invalid_search_format, invalid_search_queries = detect_invalid_search_format(result)
        row = {
            "id": src.get("id"), "question": src.get("question", ""), "answer": src.get("answer"),
            "data_source": src.get("data_source"), "split_name": src.get("split_name"),
            "decision": decision, "decision_type": decision_type,
            "result": result, "output": result,
            "has_search": has_search, "search_queries": search_queries, "search_count": len(search_queries),
            "invalid_search_format": invalid_search_format, "invalid_search_queries": invalid_search_queries,
            "invalid_search_count": len(invalid_search_queries),
            "search_action_status": "valid_search" if has_search else ("invalid_search_format" if invalid_search_format else "no_search"),
            "final_answer": final_answer,
        }
        expected = decision if expected_search is None else expected_search
        row.update(build_diagnostic_fields(expected, has_search, final_answer))
        return row

    rows = load_jsonl(args.input_jsonl)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    outputs: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        if args.mode == "direct":
            result = inference_hf_single_direct(question, model, tokenizer, DIRECT_PROMPT)
            out = build_output_row(row, result, False, "direct_baseline", expected_search=False)
        elif args.mode == "always_retrieve":
            ctx = call_retriever(question, args.retriever_url)
            prompt = f"{AGENT_PROMPT_V2_SHORT}\nYou must use the provided context in your reasoning.\n{format_context(ctx)}"
            result = inference_hf_single(question, model, tokenizer, prompt=prompt, retriever_url=args.retriever_url)
            out = build_output_row(row, result, True, "always_retrieve_baseline", expected_search=True)
            out.update({"has_search": True, "search_queries": [question], "search_count": 1, "actual_search": True})
        elif args.mode == "rule_based":
            decision = decide_retrieval(question)
            result = inference_hf_single(question, model, tokenizer, AGENT_PROMPT_V2_SHORT, args.retriever_url) if decision else inference_hf_single_direct(question, model, tokenizer, DIRECT_PROMPT)
            out = build_output_row(row, result, decision, "rule_based_baseline", expected_search=decision)
        else:
            prompt = f"{AGENT_PROMPT_V2_SHORT}\nDecide by yourself whether retrieval is needed. If needed, emit a valid <search>...</search>. If not needed, do not emit <search>."
            result = inference_hf_single(question, model, tokenizer, prompt=prompt, retriever_url=args.retriever_url)
            out = build_output_row(row, result, False, "prompt_self_routing_baseline", expected_search=False)
        outputs.append(out)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    write_jsonl(outputs, args.output_jsonl)
    print(f"Wrote {len(outputs)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
