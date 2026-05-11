from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple


SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", flags=re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
FORMAT_TAG_PATTERN = re.compile(r"</?(search|answer|think|step|reasoning|context|conclusion)>")


def extract_search_spans(text: str) -> List[Dict[str, Any]]:
    """Extract valid <search>...</search> spans and their query spans."""
    if not text:
        return []

    spans: List[Dict[str, Any]] = []
    for match in SEARCH_PATTERN.finditer(text):
        action_start, action_end = match.span(0)
        query_start, query_end = match.span(1)
        spans.append(
            {
                "action_start_char": action_start,
                "action_end_char": action_end,
                "query_start_char": query_start,
                "query_end_char": query_end,
                "query_text": match.group(1),
            }
        )
    return spans


def _tokenize_with_offsets(text: str, tokenizer):
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded.get("input_ids", [])
    offsets = encoded.get("offset_mapping", [])
    tokens = tokenizer.convert_ids_to_tokens(input_ids) if hasattr(tokenizer, "convert_ids_to_tokens") else input_ids
    return tokens, offsets


def build_token_span_mask(text: str, tokenizer, char_start: int, char_end: int) -> List[int]:
    """Return a token mask where token offsets overlap [char_start, char_end)."""
    _, offsets = _tokenize_with_offsets(text, tokenizer)
    mask = []
    for start, end in offsets:
        overlap = max(start, char_start) < min(end, char_end)
        mask.append(1 if overlap else 0)
    return mask


def build_query_token_masks(text: str, tokenizer) -> Dict[str, Any]:
    """Build action/query token masks from generated text."""
    tokens, offsets = _tokenize_with_offsets(text, tokenizer)
    n_tokens = len(offsets)
    action_mask = [0] * n_tokens
    query_token_mask = [0] * n_tokens
    search_actions: List[Dict[str, Any]] = []

    for span in extract_search_spans(text):
        action_idx = []
        query_idx = []
        for i, (start, end) in enumerate(offsets):
            if max(start, span["action_start_char"]) < min(end, span["action_end_char"]):
                action_mask[i] = 1
                action_idx.append(i)
            if max(start, span["query_start_char"]) < min(end, span["query_end_char"]):
                query_token_mask[i] = 1
                query_idx.append(i)

        search_actions.append(
            {
                "query_text": span["query_text"],
                "action_char_span": [span["action_start_char"], span["action_end_char"]],
                "query_char_span": [span["query_start_char"], span["query_end_char"]],
                "action_token_indices": action_idx,
                "query_token_indices": query_idx,
            }
        )

    return {
        "tokens": tokens,
        "offsets": offsets,
        "search_actions": search_actions,
        "action_mask": action_mask,
        "query_token_mask": query_token_mask,
    }


def extract_answer_spans(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    spans: List[Dict[str, Any]] = []
    for match in ANSWER_PATTERN.finditer(text):
        action_start, action_end = match.span(0)
        answer_start, answer_end = match.span(1)
        spans.append(
            {
                "action_start_char": action_start,
                "action_end_char": action_end,
                "answer_start_char": answer_start,
                "answer_end_char": answer_end,
                "answer_text": match.group(1),
            }
        )
    return spans


def extract_format_spans(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    return [
        {"tag_text": m.group(0), "start_char": m.start(), "end_char": m.end()}
        for m in FORMAT_TAG_PATTERN.finditer(text)
    ]


def _count_invalid_search_patterns(text: str, valid_search_count: int) -> int:
    open_exact = len(re.findall(r"<search>", text))
    close_exact = len(re.findall(r"</search>", text))
    open_like = len(re.findall(r"<search\b[^>]*>", text))
    invalid_open_like = max(open_like - open_exact, 0)
    unmatched_open = max(open_exact - valid_search_count, 0)
    unmatched_close = max(close_exact - valid_search_count, 0)
    return int(invalid_open_like + unmatched_open + unmatched_close)


def build_response_token_masks(text: str, tokenizer) -> Dict[str, Any]:
    tokens, offsets = _tokenize_with_offsets(text, tokenizer)
    n_tokens = len(offsets)
    action_mask = [0] * n_tokens
    search_query_mask = [0] * n_tokens
    answer_content_mask = [0] * n_tokens
    format_mask = [0] * n_tokens
    warnings: List[str] = []

    search_actions: List[Dict[str, Any]] = []
    for span in extract_search_spans(text):
        action_idx, query_idx = [], []
        for i, (start, end) in enumerate(offsets):
            if max(start, span["action_start_char"]) < min(end, span["action_end_char"]):
                action_mask[i] = 1
                action_idx.append(i)
            if max(start, span["query_start_char"]) < min(end, span["query_end_char"]):
                search_query_mask[i] = 1
                query_idx.append(i)
        search_actions.append({**span, "action_token_indices": action_idx, "query_token_indices": query_idx})

    answer_actions: List[Dict[str, Any]] = []
    for span in extract_answer_spans(text):
        answer_idx = []
        for i, (start, end) in enumerate(offsets):
            if max(start, span["answer_start_char"]) < min(end, span["answer_end_char"]):
                answer_content_mask[i] = 1
                answer_idx.append(i)
        answer_actions.append({**span, "answer_token_indices": answer_idx})

    format_spans = extract_format_spans(text)
    for span in format_spans:
        for i, (start, end) in enumerate(offsets):
            if max(start, span["start_char"]) < min(end, span["end_char"]):
                format_mask[i] = 1

    invalid_search_count = _count_invalid_search_patterns(text, len(search_actions))
    open_answer = len(re.findall(r"<answer>", text))
    close_answer = len(re.findall(r"</answer>", text))
    invalid_answer_count = int(max(open_answer - len(answer_actions), 0) + max(close_answer - len(answer_actions), 0))
    empty_search_count = int(sum(1 for a in search_actions if not (a.get("query_text") or "").strip()))
    empty_answer_count = int(sum(1 for a in answer_actions if not (a.get("answer_text") or "").strip()))
    if invalid_search_count > 0:
        warnings.append(f"invalid_search_count={invalid_search_count}")
    if invalid_answer_count > 0:
        warnings.append(f"invalid_answer_count={invalid_answer_count}")

    return {
        "tokens": tokens,
        "offsets": offsets,
        "search_actions": search_actions,
        "answer_actions": answer_actions,
        "format_spans": format_spans,
        "action_mask": action_mask,
        "search_query_mask": search_query_mask,
        "answer_content_mask": answer_content_mask,
        "format_mask": format_mask,
        "invalid_search_count": invalid_search_count,
        "invalid_answer_count": invalid_answer_count,
        "empty_search_count": empty_search_count,
        "empty_answer_count": empty_answer_count,
        "warnings": warnings,
    }


def allocate_reward_to_tokens(response_length: int, selected_token_indices: List[int], reward_value: float):
    """Uniformly allocate reward to selected indices with conservation."""
    scores = [0.0] * response_length
    warning = None

    if not selected_token_indices:
        warning = "empty_selected_token_indices"
        return scores, warning

    valid_indices = sorted({i for i in selected_token_indices if 0 <= i < response_length})
    if not valid_indices:
        warning = "no_valid_token_indices"
        return scores, warning

    per_token = reward_value / len(valid_indices)
    for i in valid_indices:
        scores[i] = per_token

    # numerical conservation correction (last token absorbs tiny error)
    delta = reward_value - float(sum(scores))
    if abs(delta) > 1e-12:
        scores[valid_indices[-1]] += delta

    if not math.isclose(sum(scores), reward_value, rel_tol=1e-7, abs_tol=1e-7):
        warning = "reward_conservation_error"

    return scores, warning
