# Strict Token Reward Masks

新增 `strict_query_token` 模式，并构造 `answer_content_mask` 与 `format_mask`，用于将检索 query、答案内容、结构标签分离。

“本文进一步构造 answer_content_mask 与 format_mask，将答案内容、检索 query 与结构化标签区分开来。在 strict_query_token 模式下，检索奖励仅分配至 search_query_mask，答案奖励分配至 answer_content_mask，格式奖励或惩罚分配至 format_mask。当检测到非法检索格式时，系统将该样本的 search_reward 置为 0，并将格式错误反映为 format penalty。该实现仍属于后置解析式 token-level reward allocation，不修改 PPO advantage estimation。”

- `coarse_action`: 检索奖励分配给完整 `<search>...</search>` action span。
- `query_token_uniform`: 检索奖励优先分配给 query token（允许回退）。
- `strict_query_token`: 严格分配（query/answer/format 各自独立），非法检索时 `search_reward=0`。

该实现保持 post-hoc prototype 属性，不宣称完整 BAD/POAD，也不修改 PPO advantage 估计逻辑。
