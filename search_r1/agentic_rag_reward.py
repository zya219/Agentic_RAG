"""Thesis-specific token-level reward adapter for Agentic RAG PPO training.

This module provides a minimal, runnable adapter that turns sequence-level
heuristics into token-level scores compatible with the current HiPRAG-style
PPO pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from search_r1.query_token_decomposition import allocate_reward_to_tokens, build_query_token_masks

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


def build_token_level_scores(batch, tokenizer=None, reward_decomposition_mode: str = "none") -> torch.Tensor:
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

    responses = batch.batch["responses"]
    batch_size, response_len = responses.shape
    token_scores = torch.zeros((batch_size, response_len), dtype=torch.float32, device=responses.device)

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
        if reward_decomposition_mode in {"coarse_action", "query_token_uniform"} and valid_response_len > 0:
            masks = build_query_token_masks(response_text, tokenizer)
            selected = []
            if reward_decomposition_mode == "coarse_action":
                selected = [idx for idx, v in enumerate(masks["action_mask"]) if v > 0 and idx < valid_response_len]
            elif reward_decomposition_mode == "query_token_uniform":
                selected = [idx for idx, v in enumerate(masks["query_token_mask"]) if v > 0 and idx < valid_response_len]
                if not selected:
                    selected = [idx for idx, v in enumerate(masks["action_mask"]) if v > 0 and idx < valid_response_len]
                if not selected:
                    selected = [valid_response_len - 1]

            search_only_scores, _ = allocate_reward_to_tokens(valid_response_len, selected, search_reward)
            token_scores_i = token_scores_i.clone()
            # remove default search reward assignment then overwrite with decomposition scores
            default_search_mask = torch.nonzero((search_mask > 0) if search_mask is not None else torch.zeros(valid_response_len), as_tuple=False).flatten()
            if default_search_mask.numel() > 0:
                token_scores_i[default_search_mask] -= search_reward / default_search_mask.numel()
            else:
                default_action_mask = torch.nonzero((action_mask > 0) if action_mask is not None else torch.zeros(valid_response_len), as_tuple=False).flatten()
                if default_action_mask.numel() > 0:
                    token_scores_i[default_action_mask] -= search_reward / default_action_mask.numel()
                else:
                    token_scores_i[valid_response_len - 1] -= search_reward
            token_scores_i[:valid_response_len] += torch.tensor(search_only_scores, dtype=token_scores_i.dtype, device=token_scores_i.device)
        token_scores[i, :valid_response_len] = token_scores_i.to(token_scores.device)

    return token_scores
