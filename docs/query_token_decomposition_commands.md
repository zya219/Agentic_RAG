# Commands

```bash
python debug_query_token_decomposition.py \
  --input_jsonl results/hf_reward_calibration_500_output.jsonl \
  --output_jsonl results/query_token_debug_rule_based_20.jsonl \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --max_samples 20 \
  --mode query_token_uniform
```

```bash
python debug_query_token_decomposition.py \
  --input_jsonl results/prompt_self_routing_reward_calibration_500_output.jsonl \
  --output_jsonl results/query_token_debug_prompt_self_20.jsonl \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --max_samples 20 \
  --mode query_token_uniform
```
