#!/usr/bin/env bash
set -euo pipefail

# ====== Environment-specific paths (EDIT THESE) ======
MODEL_ID="/path/to/your/qwen_or_llama_checkpoint"
TOKENIZER_ID="$MODEL_ID"
TRAIN_PARQUET="/path/to/train.parquet"
VAL_PARQUET="/path/to/val.parquet"
OUTPUT_DIR="checkpoints/agentic_rag_thesis/token_reward_proto"

# Retriever server endpoint
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

# ====== Minimal launch knobs ======
N_GPUS=1
NNODES=1
EPOCHS=1
TRAIN_BSZ=32
VAL_BSZ=32

python train_agentic_rag.py \
  --model_id "$MODEL_ID" \
  --tokenizer_id "$TOKENIZER_ID" \
  --train_parquet "$TRAIN_PARQUET" \
  --val_parquet "$VAL_PARQUET" \
  --retriever_url "$RETRIEVER_URL" \
  --output_dir "$OUTPUT_DIR" \
  --project_name "agentic_rag_thesis" \
  --experiment_name "token_reward_proto" \
  --n_gpus_per_node "$N_GPUS" \
  --nnodes "$NNODES" \
  --total_epochs "$EPOCHS" \
  --train_batch_size "$TRAIN_BSZ" \
  --val_batch_size "$VAL_BSZ"
