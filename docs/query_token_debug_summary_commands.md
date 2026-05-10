# Query-Token Debug Summary Commands

## 1) Generate three debug files

```bash
for mode in none coarse_action query_token_uniform; do
  python debug_query_token_decomposition.py \
    --input_jsonl results/prompt_self_routing_reward_calibration_500_output.jsonl \
    --output_jsonl results/query_token_debug_prompt_self_20_${mode}.jsonl \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --max_samples 20 \
    --mode ${mode}
done
```

## 2) Generate search-only case examples

```bash
python debug_query_token_decomposition.py \
  --input_jsonl results/prompt_self_routing_reward_calibration_500_output.jsonl \
  --output_jsonl results/query_token_debug_prompt_self_search_cases_20.jsonl \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --max_samples 20 \
  --mode query_token_uniform \
  --require_search
```

## 3) Summarize three modes

```bash
python summarize_query_token_debug.py \
  --inputs \
    none=results/query_token_debug_prompt_self_20_none.jsonl \
    coarse_action=results/query_token_debug_prompt_self_20_coarse_action.jsonl \
    query_token_uniform=results/query_token_debug_prompt_self_20_query_token_uniform.jsonl \
  --output_markdown results/query_token_debug_summary.md \
  --output_csv results/query_token_debug_summary.csv \
  --output_json results/query_token_debug_summary.json \
  --case_markdown results/query_token_case_examples.md \
  --max_cases 5
```
