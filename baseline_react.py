#!/usr/bin/env python3
"""Minimal ReAct-style baseline with the same retriever backend.

This is intentionally lightweight for thesis comparison runs.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import requests
from openai import OpenAI
from tqdm import tqdm

from reward import cover_exact_match
from utils import load_jsonl, write_jsonl


REACT_SYSTEM_PROMPT = """You are a QA assistant using ReAct.
Use this format repeatedly:
Thought: ...
Action: Search[query] OR Finish[final_answer]
Observation: ...

Rules:
- Use Search[...] when external facts are needed.
- Use Finish[...] to end with the final answer.
"""


def _search(query: str, retriever_url: str, topk: int = 3) -> str:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    resp = requests.post(retriever_url, json=payload, timeout=60)
    resp.raise_for_status()
    results = resp.json()["result"][0]
    passages = []
    for i, doc_item in enumerate(results):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        passages.append(f"Doc {i+1} (Title: {title}) {text}")
    return "\n".join(passages)


def react_infer_single(question: str, client: OpenAI, model_id: str, retriever_url: str, max_steps: int = 6) -> dict[str, Any]:
    transcript = [f"Question: {question}"]
    search_count = 0

    for _ in range(max_steps):
        user_content = "\n".join(transcript) + "\nRespond with one Action line."
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
        )
        text = completion.choices[0].message.content or ""
        transcript.append(text)

        finish_match = re.search(r"Finish\[(.*?)\]", text, flags=re.DOTALL)
        if finish_match:
            answer = finish_match.group(1).strip()
            return {
                "result": text,
                "final_answer": answer,
                "search_count": search_count,
                "full_trajectory": "\n".join(transcript),
            }

        search_match = re.search(r"Search\[(.*?)\]", text, flags=re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            obs = _search(query, retriever_url=retriever_url)
            transcript.append(f"Observation: {obs}")
            search_count += 1
        else:
            transcript.append("Observation: Invalid action format. Use Search[...] or Finish[...].")

    # fallback if max steps reached
    return {
        "result": transcript[-1],
        "final_answer": "",
        "search_count": search_count,
        "full_trajectory": "\n".join(transcript),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal ReAct baseline on a jsonl benchmark.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    args = parser.parse_args()

    data = load_jsonl(args.input_jsonl)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    out = []
    for row in tqdm(data):
        question = row["question"]
        pred = react_infer_single(question, client, args.model_id, args.retriever_url)
        final_answer = pred["final_answer"]
        is_correct = cover_exact_match(final_answer, row.get("golden_answers", row.get("answer", "")))
        out.append({
            **row,
            **pred,
            "baseline": "react",
            "is_correct": bool(is_correct),
        })

    write_jsonl(out, args.output_jsonl)
    print(f"Wrote {len(out)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
