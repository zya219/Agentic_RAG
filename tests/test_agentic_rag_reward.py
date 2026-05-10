import math

import torch

from search_r1.agentic_rag_reward import distribute_sequence_reward_to_token_spans


def test_distribute_sequence_reward_to_token_spans_sanity():
    response_length = 10
    answer_mask = torch.zeros(response_length, dtype=torch.float32)
    answer_mask[7:10] = 1
    search_mask = torch.zeros(response_length, dtype=torch.float32)
    search_mask[2:5] = 1
    action_mask = torch.zeros(response_length, dtype=torch.float32)
    action_mask[1:5] = 1

    scores = distribute_sequence_reward_to_token_spans(
        response_length=response_length,
        answer_reward=1.0,
        search_reward=0.5,
        action_mask=action_mask,
        answer_mask=answer_mask,
        search_mask=search_mask,
    )

    assert scores.shape[0] == response_length
    assert torch.isfinite(scores).all()

    assert torch.all(scores[7:10] > 0)
    assert torch.all(scores[2:5] > 0)

    outside_indices = [0, 5, 6]
    for idx in outside_indices:
        assert math.isclose(float(scores[idx]), 0.0, abs_tol=1e-8)


if __name__ == "__main__":
    test_distribute_sequence_reward_to_token_spans_sanity()
    print("PASS: tests/test_agentic_rag_reward.py")
