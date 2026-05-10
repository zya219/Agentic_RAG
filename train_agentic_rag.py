#!/usr/bin/env python3
"""Minimal training entry for thesis Agentic RAG PPO runs.

This script intentionally keeps configuration surface small for a one-week
prototype and launches the existing HiPRAG-style trainer with hydra overrides.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def build_cmd(args: argparse.Namespace) -> list[str]:
    tokenizer_id = args.tokenizer_id or args.model_id
    overrides = [
        f"actor_rollout_ref.model.path={args.model_id}",
        f"critic.model.path={args.model_id}",
        f"critic.model.tokenizer_path={tokenizer_id}",
        f"data.tokenizer={tokenizer_id}",
        f"data.train_files={args.train_parquet}",
        f"data.val_files={args.val_parquet}",
        f"retriever.url={args.retriever_url}",
        f"trainer.default_local_dir={args.output_dir}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        f"trainer.nnodes={args.nnodes}",
        f"trainer.total_epochs={args.total_epochs}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={args.ppo_micro_batch_size}",
        f"critic.ppo_micro_batch_size={args.critic_micro_batch_size}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_model_parallel_size}",
        f"do_search={str(args.do_search).lower()}",
        f"reward_decomposition_mode={args.reward_decomposition_mode}",
        f"format_reward_value={args.format_reward_value}",
        f"format_penalty_value={args.format_penalty_value}",
        f"reward_debug={str(args.reward_debug).lower()}",
        # Adapter usage is integrated in ray_trainer via build_token_level_scores().
        # No separate switch is needed unless you modify trainer logic.
    ]

    return [sys.executable, "-m", "verl.trainer.main_ppo_format", *overrides]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Agentic RAG thesis training entry.")

    # Essentials (edit these for your environment)
    parser.add_argument("--model_id", required=True, help="Policy/critic model path or HF id.")
    parser.add_argument("--tokenizer_id", default=None, help="Tokenizer path/id. Defaults to model_id.")
    parser.add_argument("--train_parquet", required=True, help="Training parquet path.")
    parser.add_argument("--val_parquet", required=True, help="Validation parquet path.")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve", help="Retriever endpoint.")
    parser.add_argument("--output_dir", required=True, help="Checkpoint/log output root.")

    # Lightweight runtime knobs
    parser.add_argument("--project_name", default="agentic_rag_thesis")
    parser.add_argument("--experiment_name", default="token_reward_proto")
    parser.add_argument("--n_gpus_per_node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=16)
    parser.add_argument("--ppo_micro_batch_size", type=int, default=4)
    parser.add_argument("--critic_micro_batch_size", type=int, default=4)
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--do_search", action="store_true", default=True, help="Keep Agentic RAG retrieval loop on.")
    parser.add_argument("--reward_decomposition_mode", choices=["none", "coarse_action", "query_token_uniform", "strict_query_token"], default="none")
    parser.add_argument("--format_reward_value", type=float, default=0.0)
    parser.add_argument("--format_penalty_value", type=float, default=-1.0)
    parser.add_argument("--reward_debug", action="store_true", default=False)

    parser.add_argument("--dry_run", action="store_true", help="Print command only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = build_cmd(args)
    print("Launch command:\n", " \\\n  ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
