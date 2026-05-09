# Interactive Rollout Audit (Agentic_RAG)

## Scope
This audit checks whether the current training path already supports true interactive Agentic RAG rollout (pause on `<search>`, call retriever, inject context, and continue generation) during PPO training.

## Relevant files inspected
- `inference.py`
- `search_r1/llm_agent/generation.py`
- `verl/trainer/ppo/ray_trainer.py`
- `search_r1/agentic_rag_reward.py`
- `train_agentic_rag.py`

## Findings

### 1) Where inference-time interactive retrieval happens
`inference.py` implements an explicit step loop that stops generation on `</search>`/`</answer>`, calls retriever via HTTP, appends `<context>...</context>`, then continues generation until answer completion.

### 2) Does training rollout pause generation on `<search>`?
Partially yes. In `search_r1/llm_agent/generation.py`, generated text is post-processed to stop around `</search>` or `</answer>`, then parsed into `search`/`answer` actions.

### 3) Is retrieved context injected back during training rollout?
Yes in the current generation manager path: `execute_predictions()` performs batch retrieval for search actions and injects `\n<context>...</context>\n<conclusion>` back as next observation, then `_update_rolling_state()` appends that context before the next turn.

### 4) Are token/action masks produced and consumed?
Yes, partially:
- Produced in `generation.py`: `action_mask`, `search_mask`, `answer_mask`, `step_ids`.
- Consumed by reward adapter: `search_r1/agentic_rag_reward.py` reads these masks and distributes sequence-level heuristic rewards to token spans.
- Used in trainer: `ray_trainer.py` calls `build_token_level_scores(...)` and writes `token_level_scores`.

### 5) Is `search_r1/agentic_rag_reward.py` connected to PPO trainer?
Yes. `verl/trainer/ppo/ray_trainer.py` imports and calls `build_token_level_scores(...)` in the training loop.

## Classification
**partial implementation, generation loop modification still needed**

Reason: the code already contains an interactive-style rollout manager and token-level mask scaffolding, but it still relies on heuristic parsing/distribution and does not yet demonstrate fully validated end-to-end online PPO credit assignment semantics across all cases.

## Recommended near-term route
**offline/mock retrieval route recommended** for robust reward calibration and diagnostics first, then promote to online rollout once masking/alignment and rollout reliability are validated.

## Remaining risks
- Generation must pause reliably on search action boundaries.
- Retriever server must be reachable and stable during rollout.
- Retrieved context must be consistently injected back into next-turn generation.
- Token/action masks must align exactly with generated tokenization boundaries.
- Structured reward components must be mapped to token-level scores with verified credit assignment behavior.

## Important caveat
Character spans are useful for offline debugging only; they are **not** final token-level masks. Final masks should rely on tokenizer offset mapping and/or generation-time token traces.
