# Final Test 500 Commands

## 1) Build final_test_500 subset

```bash
python build_benchmark_subset.py \
  --input_parquet ../data/HiPRAG-Dataset/test.parquet \
  --output_jsonl results/final_test_500.jsonl \
  --num_samples 500 \
  --seed 2026 \
  --split_name final_test
```

## 2) Run Direct final_test_500

```bash
python baseline_experiments.py --mode direct --input_jsonl results/final_test_500.jsonl --output_jsonl results/direct_reward_final_test_500_output.jsonl
```

## 3) Run Always Retrieve final_test_500

```bash
python baseline_experiments.py --mode always_retrieve --input_jsonl results/final_test_500.jsonl --output_jsonl results/always_retrieve_reward_final_test_500_output.jsonl
```

## 4) Run Rule-based / Current Agentic final_test_500

```bash
python baseline_experiments.py --mode hf_reward --input_jsonl results/final_test_500.jsonl --output_jsonl results/hf_reward_final_test_500_output.jsonl
```

## 5) Run Prompt Self-Routing final_test_500

```bash
python baseline_experiments.py --mode prompt_self_routing --input_jsonl results/final_test_500.jsonl --output_jsonl results/prompt_self_routing_reward_final_test_500_output.jsonl
```

## 6) Evaluate each output with `evaluate_predictions.py`

```bash
python evaluate_predictions.py --prediction_jsonl results/direct_reward_final_test_500_output.jsonl --summary_json results/direct_reward_final_test_500_summary.json
python evaluate_predictions.py --prediction_jsonl results/always_retrieve_reward_final_test_500_output.jsonl --summary_json results/always_retrieve_reward_final_test_500_summary.json
python evaluate_predictions.py --prediction_jsonl results/hf_reward_final_test_500_output.jsonl --summary_json results/hf_reward_final_test_500_summary.json
python evaluate_predictions.py --prediction_jsonl results/prompt_self_routing_reward_final_test_500_output.jsonl --summary_json results/prompt_self_routing_reward_final_test_500_summary.json
```

## 7) Aggregate summaries

```bash
python aggregate_baseline_results.py \
  --inputs \
    direct=results/direct_reward_final_test_500_summary.json \
    always_retrieve=results/always_retrieve_reward_final_test_500_summary.json \
    rule_based=results/hf_reward_final_test_500_summary.json \
    prompt_self_routing=results/prompt_self_routing_reward_final_test_500_summary.json \
  --output_markdown results/final_test_500_baseline_comparison.md \
  --output_csv results/final_test_500_baseline_comparison.csv \
  --output_json results/final_test_500_baseline_comparison.json \
  --title "Final-Test-500 Baseline Comparison"
```

## 8) Warning

Do **not** modify prompt, reward, routing rules, or baseline logic after looking at final_test results.
