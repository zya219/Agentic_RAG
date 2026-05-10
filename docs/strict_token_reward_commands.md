# Strict Token Reward Commands

```bash
python visualize_token_rewards.py \
  --input_jsonl results/prompt_self_routing_reward_calibration_500_output.jsonl \
  --output_markdown results/token_reward_cases_prompt_self.md \
  --output_jsonl results/token_reward_cases_prompt_self.jsonl \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --max_samples 5 \
  --mode strict_query_token \
  --require_search
```

```bash
python summarize_token_reward_debug.py \
  --input_jsonl results/token_reward_cases_prompt_self.jsonl \
  --output_markdown results/token_reward_debug_summary.md \
  --output_csv results/token_reward_debug_summary.csv \
  --output_json results/token_reward_debug_summary.json
```
