"""Thesis-specific token-level reward adapter for Agentic RAG PPO training.

This module provides a minimal, runnable adapter that turns sequence-level
heuristics into token-level scores compatible with the current HiPRAG-style
PPO pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from search_r1.query_token_decomposition import allocate_reward_to_tokens, build_query_token_masks, build_response_token_masks

import torch

from reward import cover_exact_match


def extract_final_answer_from_response(response_text: str) -> str:
    """Extract the final answer content from ``<answer>...</answer>``.

    Returns an empty string when no valid answer span is found.
    """
    if not response_text:
        return ""
    matches = re.findall(r"<answer>(.*?)</answer>", response_text, flags=re.DOTALL)
    return matches[-1].strip() if matches else ""


def compute_answer_correctness_reward(response_text: str, ground_truth: Any) -> float:
    """Compute answer correctness reward.

    Returns:
        +1.0 when final answer matches ground truth, else 0.0.
    """
    predicted = extract_final_answer_from_response(response_text)
    if not predicted:
        return 0.0
    return 1.0 if cover_exact_match(predicted, ground_truth) else 0.0


def compute_search_behavior_reward(response_text: str) -> float:
    """Compute heuristic search behavior reward.

    Reward design (heuristic):
      - small positive reward for each non-empty search query
      - penalty for malformed search tags
      - penalty for duplicate (redundant) queries
    """
    if not response_text:
        return 0.0

    search_open = response_text.count("<search>")
    search_close = response_text.count("</search>")
    malformed = abs(search_open - search_close)

    queries = [q.strip() for q in re.findall(r"<search>(.*?)</search>", response_text, flags=re.DOTALL)]
    valid_queries = [q for q in queries if len(q) >= 2]

    seen = set()
    duplicate_count = 0
    for q in valid_queries:
        normalized = re.sub(r"\s+", " ", q.lower())
        if normalized in seen:
            duplicate_count += 1
        else:
            seen.add(normalized)

    reward = 0.0
    reward += 0.05 * len(valid_queries)
    reward -= 0.10 * malformed
    reward -= 0.05 * duplicate_count

    return float(reward)


def distribute_sequence_reward_to_token_spans(
    response_length: int,
    answer_reward: float,
    search_reward: float,
    action_mask: torch.Tensor | None = None,
    answer_mask: torch.Tensor | None = None,
    search_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distribute sequence-level rewards to token spans.

    Main credit assignment:
      - Answer correctness goes to ``answer_mask`` tokens (fallback: last token).
      - Search behavior reward goes to ``search_mask`` tokens (fallback: action tokens).
      - Non-action reasoning tokens receive no direct action reward by default.
    """
    scores = torch.zeros(response_length, dtype=torch.float32)

    action_mask = action_mask if action_mask is not None else torch.zeros(response_length, dtype=torch.float32)
    answer_mask = answer_mask if answer_mask is not None else torch.zeros(response_length, dtype=torch.float32)
    search_mask = search_mask if search_mask is not None else torch.zeros(response_length, dtype=torch.float32)

    answer_idx = torch.nonzero(answer_mask > 0, as_tuple=False).flatten()
    if answer_idx.numel() > 0:
        scores[answer_idx] += answer_reward / answer_idx.numel()
    elif response_length > 0:
        scores[response_length - 1] += answer_reward

    search_idx = torch.nonzero(search_mask > 0, as_tuple=False).flatten()
    if search_idx.numel() > 0:
        scores[search_idx] += search_reward / search_idx.numel()
    else:
        action_idx = torch.nonzero(action_mask > 0, as_tuple=False).flatten()
        if action_idx.numel() > 0:
            scores[action_idx] += search_reward / action_idx.numel()
        elif response_length > 0:
            scores[response_length - 1] += search_reward

    return scores


def _extract_ground_truth(non_tensor_batch: Dict[str, Any]) -> Any:
    """Extract ground-truth answers from common data layouts used in this repo."""
    rm = non_tensor_batch.get("reward_model", {}) if isinstance(non_tensor_batch, dict) else {}
    gt = rm.get("ground_truth")

    if isinstance(gt, dict):
        if "target" in gt:
            return gt["target"]
        if "answer" in gt:
            return gt["answer"]
    if gt is not None:
        return gt

    for key in ["golden_answers", "answer", "target"]:
        if isinstance(non_tensor_batch, dict) and key in non_tensor_batch:
            return non_tensor_batch[key]

    return ""




def build_token_level_scores_with_debug(
    batch,
    tokenizer=None,
    reward_decomposition_mode: str = "none",
    format_reward_value: float = 0.0,
    format_penalty_value: float = -1.0,
    search_cost_value: float = 0.1,
    repeat_search_penalty_value: float = 0.2,
    answer_missing_penalty_value: float = -1.0,
):
    token_scores, debug_info = build_token_level_scores(
        batch=batch,
        tokenizer=tokenizer,
        reward_decomposition_mode=reward_decomposition_mode,
        format_reward_value=format_reward_value,
        format_penalty_value=format_penalty_value,
        search_cost_value=search_cost_value,
        repeat_search_penalty_value=repeat_search_penalty_value,
        answer_missing_penalty_value=answer_missing_penalty_value,
        return_debug=True,
    )
    return token_scores, debug_info

def build_token_level_scores(batch, tokenizer=None, reward_decomposition_mode: str = "none", format_reward_value: float = 0.0, format_penalty_value: float = -1.0, search_cost_value: float = 0.1, repeat_search_penalty_value: float = 0.2, answer_missing_penalty_value: float = -1.0, return_debug: bool = False):
    """Build PPO-compatible token-level scores from a rollout batch.

    Args:
        batch: DataProto-like batch with ``batch`` and ``non_tensor_batch`` fields.
        tokenizer: Optional tokenizer override. If omitted, attempts to read
            ``batch.meta_info['tokenizer']`` (if present).

    Returns:
        Tensor of shape ``[batch_size, response_len]`` with token-level scores.
    """
    if tokenizer is None:
        tokenizer = getattr(batch, "meta_info", {}).get("tokenizer", None)
    if tokenizer is None:
        raise ValueError("tokenizer is required for build_token_level_scores")

    allowed_modes = {"none", "coarse_action", "query_token_uniform", "strict_query_token", "strict_query_token_cost"}
    if reward_decomposition_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported reward_decomposition_mode={reward_decomposition_mode!r}; "
            f"expected one of {sorted(allowed_modes)}"
        )

    responses = batch.batch["responses"]
    batch_size, response_len = responses.shape
    token_scores = torch.zeros((batch_size, response_len), dtype=torch.float32, device=responses.device)
    debug_rows = []

    has_action_mask = "action_mask" in batch.batch
    has_answer_mask = "answer_mask" in batch.batch
    has_search_mask = "search_mask" in batch.batch

    for i in range(len(batch)):
        item = batch[i]
        prompt_len = int(item.batch["prompts"].shape[-1]) if "prompts" in item.batch else 0

        attn = item.batch["attention_mask"]
        valid_response_len = int(attn[prompt_len:].sum().item()) if prompt_len > 0 else int(attn.sum().item())
        valid_response_len = min(valid_response_len, response_len)

        response_ids = item.batch["responses"][:valid_response_len]
        response_text = tokenizer.decode(response_ids)

        ground_truth = _extract_ground_truth(item.non_tensor_batch)

        answer_reward = compute_answer_correctness_reward(response_text, ground_truth)
        search_reward = compute_search_behavior_reward(response_text)

        action_mask = item.batch["action_mask"][-response_len:][:valid_response_len].float() if has_action_mask else None
        answer_mask = item.batch["answer_mask"][-response_len:][:valid_response_len].float() if has_answer_mask else None
        search_mask = item.batch["search_mask"][-response_len:][:valid_response_len].float() if has_search_mask else None

        token_scores_i = distribute_sequence_reward_to_token_spans(
            response_length=valid_response_len,
            answer_reward=answer_reward,
            search_reward=search_reward,
            action_mask=action_mask,
            answer_mask=answer_mask,
            search_mask=search_mask,
        )
        warnings = []
        if reward_decomposition_mode in {"coarse_action", "query_token_uniform", "strict_query_token", "strict_query_token_cost"} and valid_response_len > 0:
            masks = build_response_token_masks(response_text, tokenizer)
            token_scores_i = token_scores_i.clone()
            # clear default search allocation
            default_search_mask = torch.nonzero((search_mask > 0) if search_mask is not None else torch.zeros(valid_response_len), as_tuple=False).flatten()
            if default_search_mask.numel() > 0:
                token_scores_i[default_search_mask] -= search_reward / default_search_mask.numel()
            else:
                default_action_mask = torch.nonzero((action_mask > 0) if action_mask is not None else torch.zeros(valid_response_len), as_tuple=False).flatten()
                if default_action_mask.numel() > 0:
                    token_scores_i[default_action_mask] -= search_reward / default_action_mask.numel()
                elif valid_response_len > 0:
                    token_scores_i[valid_response_len - 1] -= search_reward

            search_selected = []
            if reward_decomposition_mode == "coarse_action":
                search_selected = [idx for idx, v in enumerate(masks["action_mask"]) if v > 0 and idx < valid_response_len]
            else:
                search_selected = [idx for idx, v in enumerate(masks["search_query_mask"]) if v > 0 and idx < valid_response_len]
                if reward_decomposition_mode == "query_token_uniform" and not search_selected:
                    search_selected = [idx for idx, v in enumerate(masks["action_mask"]) if v > 0 and idx < valid_response_len] or [valid_response_len - 1]

            strict_search_reward = search_reward
            if reward_decomposition_mode == "strict_query_token" and masks["invalid_search_count"] > 0:
                strict_search_reward = 0.0
            if reward_decomposition_mode == "strict_query_token" and not search_selected:
                strict_search_reward = 0.0
                warnings.append("no_query_tokens_for_strict_search_reward")
            search_scores = [0.0] * valid_response_len
            if reward_decomposition_mode == "strict_query_token_cost":
                # cost-aware post-hoc query-token-level reward decomposition:
                # still uses strict post-hoc parsing masks (not online POAD / not action-space changes).
                valid_search_actions = [
                    a for a in masks.get("search_actions", [])
                    if a.get("query_token_indices") and (a.get("query_text") or "").strip()
                ]
                for si, action in enumerate(valid_search_actions):
                    qidx = [idx for idx in action["query_token_indices"] if idx < valid_response_len]
                    if not qidx:
                        continue
                    pos_scores, _ = allocate_reward_to_tokens(valid_response_len, qidx, float(search_reward))
                    cost_scores, _ = allocate_reward_to_tokens(valid_response_len, qidx, -float(search_cost_value))
                    rep_scores = [0.0] * valid_response_len
                    if si >= 1:
                        rep_scores, _ = allocate_reward_to_tokens(valid_response_len, qidx, -float(repeat_search_penalty_value))
                    search_scores = [x + y + z + w for x, y, z, w in zip(search_scores, pos_scores, cost_scores, rep_scores)]
                token_scores_i[:valid_response_len] += torch.tensor(search_scores, dtype=token_scores_i.dtype, device=token_scores_i.device)
            else:
                search_scores, _ = allocate_reward_to_tokens(valid_response_len, search_selected, strict_search_reward)
                token_scores_i[:valid_response_len] += torch.tensor(search_scores, dtype=token_scores_i.dtype, device=token_scores_i.device)

            if reward_decomposition_mode in {"strict_query_token", "strict_query_token_cost"}:
                # remove default answer assignment and re-allocate by answer mask
                answer_idx_default = torch.nonzero((answer_mask > 0) if answer_mask is not None else torch.zeros(valid_response_len), as_tuple=False).flatten()
                if answer_idx_default.numel() > 0:
                    token_scores_i[answer_idx_default] -= answer_reward / answer_idx_default.numel()
                elif valid_response_len > 0:
                    token_scores_i[valid_response_len - 1] -= answer_reward
                ans_selected = [idx for idx, v in enumerate(masks["answer_content_mask"]) if v > 0 and idx < valid_response_len]
                has_valid_answer = any((a.get("answer_text") or "").strip() for a in masks.get("answer_actions", []))
                ans_scores, _ = allocate_reward_to_tokens(valid_response_len, ans_selected, answer_reward if has_valid_answer else 0.0)
                token_scores_i[:valid_response_len] += torch.tensor(ans_scores, dtype=token_scores_i.dtype, device=token_scores_i.device)

                format_selected = [idx for idx, v in enumerate(masks["format_mask"]) if v > 0 and idx < valid_response_len]
                invalid_format = (masks["invalid_search_count"] > 0) or (masks.get("invalid_answer_count", 0) > 0) or (masks.get("empty_search_count", 0) > 0)
                format_reward = format_reward_value if not invalid_format else format_penalty_value
                if not format_selected and invalid_format and valid_response_len > 0:
                    format_selected = [valid_response_len - 1]
                fmt_scores, _ = allocate_reward_to_tokens(valid_response_len, format_selected, format_reward)
                token_scores_i[:valid_response_len] += torch.tensor(fmt_scores, dtype=token_scores_i.dtype, device=token_scores_i.device)
                if reward_decomposition_mode == "strict_query_token_cost" and not has_valid_answer:
                    answer_missing_selected = format_selected or ([valid_response_len - 1] if valid_response_len > 0 else [])
                    miss_scores, _ = allocate_reward_to_tokens(valid_response_len, answer_missing_selected, float(answer_missing_penalty_value))
                    token_scores_i[:valid_response_len] += torch.tensor(miss_scores, dtype=token_scores_i.dtype, device=token_scores_i.device)
            debug_rows.append({
                "query_token_count": int(sum(masks["search_query_mask"][:valid_response_len])),
                "answer_content_token_count": int(sum(masks["answer_content_mask"][:valid_response_len])),
                "format_token_count": int(sum(masks["format_mask"][:valid_response_len])),
                "invalid_search_count": int(masks["invalid_search_count"]),
                "invalid_format_count": int((masks["invalid_search_count"] > 0) or (masks.get("invalid_answer_count", 0) > 0) or (masks.get("empty_search_count", 0) > 0)),
                "answer_missing_count": int(not any((a.get("answer_text") or "").strip() for a in masks.get("answer_actions", []))),
                "search_count": int(sum(1 for a in masks.get("search_actions", []) if (a.get("query_text") or "").strip())),
                "repeated_search_count": int(max(0, sum(1 for a in masks.get("search_actions", []) if (a.get("query_text") or "").strip()) - 1)),
                "search_reward_sum": float(sum(search_scores)),
                "answer_reward_sum": float(sum(ans_scores) if 'ans_scores' in locals() else answer_reward),
                "format_reward_sum": float(sum(fmt_scores) if 'fmt_scores' in locals() else 0.0),
                "total_token_score_sum": float(token_scores_i[:valid_response_len].sum().item()),
                "search_query_mask": masks["search_query_mask"][:valid_response_len],
                "answer_content_mask": masks["answer_content_mask"][:valid_response_len],
                "format_mask": masks["format_mask"][:valid_response_len],
                "action_mask": masks["action_mask"][:valid_response_len],
                "warnings": list(masks.get("warnings", [])) + warnings,
            })

        token_scores[i, :valid_response_len] = token_scores_i.to(token_scores.device)

    return (token_scores, debug_rows) if return_debug else token_scores
