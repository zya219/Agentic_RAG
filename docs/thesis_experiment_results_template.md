# Thesis Experiment Results Template

## 1. Calibration-500 Baseline Comparison

| Method | EM | F1 | Avg Search | Search Mismatch | Invalid Search Format | Answer Missing | Avg Answer Reward | Avg Final Reward | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct |  |  |  |  |  |  |  |  |  |
| Always Retrieve |  |  |  |  |  |  |  |  |  |
| Rule-based / Current Agentic |  |  |  |  |  |  |  |  |  |
| Prompt Self-Routing |  |  |  |  |  |  |  |  |  |

## 2. Final-Test-500 Baseline Comparison

| Method | EM | F1 | Avg Search | Search Mismatch | Invalid Search Format | Answer Missing | Avg Answer Reward | Avg Final Reward | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct |  |  |  |  |  |  |  |  |  |
| Always Retrieve |  |  |  |  |  |  |  |  |  |
| Rule-based / Current Agentic |  |  |  |  |  |  |  |  |  |
| Prompt Self-Routing |  |  |  |  |  |  |  |  |  |

## 3. Reward Component Analysis

| Method | Answer Reward | Search Decision Reward | Format Reward | Efficiency Penalty | Answer Missing Penalty | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Direct |  |  |  |  |  |  |
| Always Retrieve |  |  |  |  |  |  |
| Rule-based / Current Agentic |  |  |  |  |  |  |
| Prompt Self-Routing |  |  |  |  |  |  |

## 4. Suggested interpretation prompts (中文)

- “Direct baseline 用于衡量不检索时模型仅依赖参数知识的表现。”
- “Always Retrieve baseline 用于衡量无条件检索是否带来收益或噪声。”
- “Rule-based Routing 表示外部启发式检索决策。”
- “Prompt Self-Routing 表示未经强化学习训练时，模型自身基于 prompt 的检索决策能力。”
- “若 Always Retrieve 的 final_reward 较高但 answer_f1 未提升，应说明 final_reward 受到 search_decision_reward 的影响，不应单独作为答案质量指标。”
