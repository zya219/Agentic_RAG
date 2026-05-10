# Query-Token-Level Action Decomposition (Post-hoc)

## Motivation
Current Agentic RAG reward assignment can place search reward coarsely on broad action spans. Query-token-level decomposition improves interpretability by assigning search reward specifically to query tokens inside `<search>query</search>`.

## Method
1. Parse generated trajectories to extract valid `<search>query</search>` spans.
2. Use tokenizer `offset_mapping` to align character spans to token indices.
3. Build:
   - `action_mask`: tokens inside `<search>...</search>`
   - `query_token_mask`: tokens inside query content only
4. Allocate search reward with reward conservation:
   - token-level score sum equals original search reward component.

## Modes
- `none`: preserve existing behavior (default).
- `coarse_action`: distribute search reward uniformly over full `<search>...</search>` action tokens.
- `query_token_uniform`: distribute search reward uniformly over query tokens only; fallback safely to `coarse_action` or last token when needed.

## Thesis-safe wording (中文)
“本文在现有 Agentic RAG 系统上实现后置解析式 query-token-level action decomposition。系统通过 tokenizer offset mapping 将 `<search>query</search>` 中的 query 内容定位到 token 级别，并将检索行为 reward 按 reward conservation 原则分配给每个 query token，从而构造 PPO-compatible token_level_scores。”

## Limitations
- This is **post-hoc decomposition**.
- It does **not** implement full BAD-style Bellman backup.
- It does **not** modify PPO advantage estimation.
- It is an interpretable POAD-style reward allocation prototype.
