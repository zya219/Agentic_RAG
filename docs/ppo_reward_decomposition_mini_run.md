# PPO Reward Decomposition Mini-Run

> This is a **training-chain mini-run**, not a full performance training run.

## 1) Dry-run command (strict_query_token)

```bash
python train_agentic_rag.py \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --train_parquet ../data/HiPRAG-Dataset/train.parquet \
  --val_parquet ../data/HiPRAG-Dataset/train.parquet \
  --output_dir runs/mini_run_strict_query_token_dryrun \
  --reward_decomposition_mode strict_query_token \
  --format_reward_value 0.0 \
  --format_penalty_value -1.0 \
  --total_epochs 1 \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --ppo_mini_batch_size 1 \
  --ppo_micro_batch_size 1 \
  --critic_micro_batch_size 1 \
  --tensor_model_parallel_size 1 \
  --reward_debug \
  --dry_run
```

## 2) Tiny non-dry-run command (strict_query_token)

```bash
python train_agentic_rag.py \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --train_parquet ../data/HiPRAG-Dataset/train.parquet \
  --val_parquet ../data/HiPRAG-Dataset/train.parquet \
  --output_dir runs/mini_run_strict_query_token \
  --reward_decomposition_mode strict_query_token \
  --format_reward_value 0.0 \
  --format_penalty_value -1.0 \
  --total_epochs 1 \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --ppo_mini_batch_size 1 \
  --ppo_micro_batch_size 1 \
  --critic_micro_batch_size 1 \
  --tensor_model_parallel_size 1 \
  --reward_debug
```

## 3) Four-mode mini-run loop

```bash
for mode in none coarse_action query_token_uniform strict_query_token; do
  python train_agentic_rag.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --train_parquet ../data/HiPRAG-Dataset/train.parquet \
    --val_parquet ../data/HiPRAG-Dataset/train.parquet \
    --output_dir runs/mini_run_${mode} \
    --reward_decomposition_mode ${mode} \
    --format_reward_value 0.0 \
    --format_penalty_value -1.0 \
    --total_epochs 1 \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --ppo_mini_batch_size 1 \
    --ppo_micro_batch_size 1 \
    --critic_micro_batch_size 1 \
    --tensor_model_parallel_size 1 \
    --reward_debug
done
```
