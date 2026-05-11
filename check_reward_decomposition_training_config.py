#!/usr/bin/env python3
"""Smoke test for reward decomposition CLI -> hydra override propagation."""

from __future__ import annotations

import subprocess
import sys

MODES = ["none", "coarse_action", "query_token_uniform", "strict_query_token", "strict_query_token_cost"]


def run_mode(mode: str) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "train_agentic_rag.py",
        "--model_id", "Qwen/Qwen2.5-3B-Instruct",
        "--train_parquet", "../data/HiPRAG-Dataset/train.parquet",
        "--val_parquet", "../data/HiPRAG-Dataset/train.parquet",
        "--output_dir", f"runs/test_reward_mode_dryrun_{mode}",
        "--reward_decomposition_mode", mode,
        "--reward_debug",
        "--dry_run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    required = [
        f"reward_decomposition_mode={mode}",
        "format_reward_value=",
        "format_penalty_value=",
        "reward_debug=",
    ]
    ok = proc.returncode == 0 and all(token in out for token in required)
    return ok, out


def main() -> int:
    failures = []
    for mode in MODES:
        ok, out = run_mode(mode)
        if ok:
            print(f"[PASS] {mode}")
        else:
            print(f"[FAIL] {mode}")
            failures.append((mode, out))

    if failures:
        for mode, out in failures:
            print(f"\n--- output for mode={mode} ---")
            print(out)
        return 1
    print("All reward decomposition config checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
