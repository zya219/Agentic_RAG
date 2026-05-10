from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple


SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", flags=re.DOTALL)


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
