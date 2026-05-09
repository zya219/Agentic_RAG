# AutoDL Smoke Test Workflow

## Directory layout
- `/root/autodl-tmp/ARAG/Agentic_RAG`
- `/root/autodl-tmp/ARAG/HiPRAG`
- `/root/autodl-tmp/ARAG/ADRL`
- `/root/autodl-tmp/ARAG/data`

## Environment
- Conda env: `arag`

## Launch retrieval server
```bash
cd /root/autodl-tmp/ARAG/Agentic_RAG
conda activate arag
bash retrieval_launch.sh
```

## Build dev benchmark subset (from train.parquet)
```bash
python build_benchmark_subset.py \
  --input_parquet ../data/HiPRAG-Dataset/train.parquet \
  --output_jsonl results/benchmark_dev_100.jsonl \
  --num_samples 100 \
  --split_name dev
```

## Run HF inference on dev subset
```bash
python inference.py \
  --input_jsonl results/benchmark_dev_100.jsonl \
  --output_jsonl results/hf_benchmark_dev_100_output.jsonl \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --retriever_url http://127.0.0.1:8000/retrieve
```

## Evaluate predictions (deterministic + structured reward components)
```bash
python evaluate_predictions.py \
  --pred_jsonl results/hf_benchmark_dev_100_output.jsonl \
  --gold_jsonl results/benchmark_dev_100.jsonl \
  --output_jsonl results/hf_benchmark_dev_100_eval.jsonl \
  --summary_json results/hf_benchmark_dev_100_summary.json
```

## Build final test subset (final evaluation only)
```bash
python build_benchmark_subset.py \
  --input_parquet ../data/HiPRAG-Dataset/test.parquet \
  --output_jsonl results/benchmark_test_100.jsonl \
  --num_samples 100 \
  --split_name test
```

## Analyze inference results
```bash
python analyze_inference_results.py \
  --input_jsonl results/hf_test_output.jsonl \
  --summary_json results/hf_test_summary.json \
  --mismatch_jsonl results/search_mismatch_cases.jsonl
```

## Analyze manual eval results
```bash
python analyze_manual_eval.py \
  --input_jsonl results/manual_eval_10case.jsonl \
  --summary_json results/manual_eval_summary_10case.json \
  --problem_jsonl results/manual_eval_problem_cases_10case.jsonl
```

## Notes
- The current retrieval server uses CPU FAISS.
- `e5_Flat.index` is ~61GB, while a single RTX 4090 has 24GB VRAM.
- `<search>...</search>` is the only valid search action format.
- HiPRAG-Dataset commonly uses `golden_answers` as the gold answer field in parquet; `build_benchmark_subset.py` auto-detects and exports it as `answer`.
- `<search query="...">...</search>` (or any `<search ...>...</search>` with attributes) is detected as `invalid_search_format`, not counted as a valid search action.
- This distinction is kept for future reward design and token-level credit assignment.
- **Do not tune reward weights on `test.parquet`; use `test.parquet` only for final evaluation.**
