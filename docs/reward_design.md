# Reward Design Notes (Deterministic + Structured)

## Why EM/F1 alone is insufficient
Open-ended LLM outputs may be semantically correct but lexically different, include extra explanation, or use alternate phrasing. EM/F1 alone cannot capture all helpful retrieval behavior and action-format quality.

## Why normalized EM/F1 is still useful
Normalized EM/F1 is deterministic, reproducible, cheap, and easy to compare across runs. It provides stable automatic signals for regression tracking and baseline reward calibration.

## Why no LLM-as-a-Judge in PPO inner loop
PPO inner-loop reward must be fast, deterministic, and operationally stable. External judge calls add latency/cost/variance and complicate reproducibility.

## Why structured reward components are needed
A single scalar reward hides behavior details. Structured components expose answer quality, search decision quality, format compliance, and efficiency so failures can be diagnosed and later mapped to token/action masks.

## Search behavior states
- `no_search`: no valid `<search>...</search>` used.
- `valid_search`: at least one valid `<search>...</search>` action.
- `invalid_search_format`: malformed search attempt like `<search ...>...</search>` (attributes/invalid format).

## Bridge to future token/action masks
Planned mapping:
- `answer_reward` -> answer tokens (`answer_mask`)
- `search_decision_reward` -> search action tokens (`action_mask`/`search_mask`)
- `format_reward` -> XML tag tokens (format mask)
- `efficiency_penalty` -> search action count (action-level signal)

## Trajectory segments vs real token masks
`trajectory_segments` store extracted text spans for offline analysis. Optional char spans are debug artifacts only and are **not** final token-level masks. Real token masks should be produced from tokenizer offset mapping or generation-time token traces.

## Split policy warning
Calibrate/tune reward weights on train/dev only. Do **not** tune reward weights on test split; use test only for final evaluation.
