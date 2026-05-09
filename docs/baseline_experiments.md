# Baseline / Ablation Experiments

This document describes reproducible baseline runners for `Agentic_RAG`.

## Baseline modes

`baseline_experiments.py` supports four modes via `--mode`:

1. `direct`
- Fixed strategy baseline: never retrieves.
- The model answers directly.
- Useful as a lower-bound cost baseline.

2. `always_retrieve`
- Fixed strategy baseline: always retrieves once using the question as query.
- Useful as an upper-bound retrieval-usage baseline.

3. `rule_based`
- Reuses current external heuristic routing (`decide_retrieval`) from `inference.py`.
- This is an ablation of routing policy, not model-internal decision making.
- Represents current Agentic / decision-aware baseline.

4. `prompt_self_routing`
- No external Python routing.
- The model decides whether to emit `<search>...</search>` based on prompt instructions.
- This is a fairer non-RL model-internal routing baseline.

## Why these baselines matter

- Direct and Always Retrieve define fixed-strategy anchor points.
- Rule-based isolates the effect of heuristic external routing.
- Prompt self-routing isolates model-internal tool-use behavior without PPO/RL integration.

These baselines provide clean comparison data for the current POAD-style/token-level reward prototype narrative, without claiming full ADRL/POAD/BAD integration.

## Run baselines

```bash
python baseline_experiments.py \
  --input_jsonl results/reward_calibration_500.jsonl \
  --output_jsonl results/direct_reward_calibration_500_output.jsonl \
  --mode direct \
  --model_id Qwen/Qwen2.5-3B-Instruct
```

```bash
python baseline_experiments.py \
  --input_jsonl results/reward_calibration_500.jsonl \
  --output_jsonl results/always_retrieve_reward_calibration_500_output.jsonl \
  --mode always_retrieve \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --retriever_url http://127.0.0.1:8000/retrieve
```

```bash
python baseline_experiments.py \
  --input_jsonl results/reward_calibration_500.jsonl \
  --output_jsonl results/rule_based_reward_calibration_500_output.jsonl \
  --mode rule_based \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --retriever_url http://127.0.0.1:8000/retrieve
```

```bash
python baseline_experiments.py \
  --input_jsonl results/reward_calibration_500.jsonl \
  --output_jsonl results/prompt_self_routing_reward_calibration_500_output.jsonl \
  --mode prompt_self_routing \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --retriever_url http://127.0.0.1:8000/retrieve
```

## Evaluate outputs

```bash
python evaluate_predictions.py \
  --pred_jsonl results/direct_reward_calibration_500_output.jsonl \
  --gold_jsonl results/reward_calibration_500.jsonl \
  --output_jsonl results/direct_reward_calibration_500_eval.jsonl \
  --summary_json results/direct_reward_calibration_500_summary.json
```

```bash
python evaluate_predictions.py \
  --pred_jsonl results/always_retrieve_reward_calibration_500_output.jsonl \
  --gold_jsonl results/reward_calibration_500.jsonl \
  --output_jsonl results/always_retrieve_reward_calibration_500_eval.jsonl \
  --summary_json results/always_retrieve_reward_calibration_500_summary.json
```

## Data split warning

- `train.parquet` derived subsets (dev/calibration) are valid for development/tuning.
- `test.parquet` must be reserved for final evaluation only.
