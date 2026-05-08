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

## Run HF inference
```bash
cd /root/autodl-tmp/ARAG/Agentic_RAG
conda activate arag
python inference.py \
  --input_jsonl results/test_template.jsonl \
  --output_jsonl results/hf_test_output.jsonl \
  --model_id Qwen/Qwen2.5-3B-Instruct
```

## Analyze inference results
```bash
python analyze_inference_results.py \
  --input_jsonl results/hf_test_output.jsonl \
  --summary_json results/hf_test_summary.json \
  --mismatch_jsonl results/search_mismatch_cases.jsonl
```

## Notes
- The current retrieval server uses CPU FAISS.
- `e5_Flat.index` is ~61GB, while a single RTX 4090 has 24GB VRAM.

## Analyze manual eval results
```bash
python analyze_manual_eval.py \
  --input_jsonl results/manual_eval_10case.jsonl \
  --summary_json results/manual_eval_summary_10case.json \
  --problem_jsonl results/manual_eval_problem_cases_10case.jsonl
```

