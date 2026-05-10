# Thesis Method Positioning

## 1. Thesis Method Name

**POAD-style Token-level Reward Agentic RAG**

## 2. What is implemented

- Agentic RAG inference with `<search>...</search>` style retrieval actions.
- Baseline experiment pipelines including direct, retrieval-oriented, rule-based, and prompt-self-routing variants.
- Structured evaluation using EM/F1 and reward-related summary metrics.
- Token-level reward scaffold using `action_mask`, `search_mask`, and `answer_mask`.
- PPO training entry dry-run path through `train_agentic_rag.py`.

## 3. What is not implemented

- This is **not** a full reproduction of ADRL/POAD/BAD.
- The current implementation does **not** modify PPO advantage estimation.
- BAD-style Bellman backup is not fully implemented.
- Full online interactive PPO training remains future work.

## 4. Why this is still valid for the thesis

- The thesis target is token-level action decomposition for Agentic RAG strategy optimization.
- The implementation already maps search behavior into token/span-level structured actions.
- The reward adapter maps answer/search/action-level signals to token regions.
- Baseline experiments provide evidence for and against different retrieval strategies, supporting thesis analysis.

## 5. Suggested wording for thesis

“本文借鉴 POAD 的 token-level action decomposition 思想，将 Agentic RAG 中的检索行为建模为由 `<search>`、query tokens 与 `</search>` 组成的 token-level action sequence。系统进一步设计 structured reward components，并通过 search_mask、action_mask 与 answer_mask 将检索行为奖励、格式约束和答案正确性信号映射到不同 token 区域，形成 POAD-style token-level reward adapter。当前实现不修改 PPO 的 advantage estimation 公式，而是提供可接入 PPO 框架的 token-level score prototype；完整 BAD-style PPO 更新作为后续工作。”

## 6. Suggested limitations

- Exact EM/F1 may underestimate semantically correct but lexically different long answers.
- Forced retrieval methods can obtain fixed `search_decision_reward`, inflating `avg_final_reward`.
- Final test split results must not be used for reward or routing tuning.
- Full POAD/BAD implementation remains future work.
