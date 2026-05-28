# Final-Test-500 Baseline Comparison

> Warning: avg_final_reward may not be directly comparable across baselines if search_decision_reward gives fixed reward to forced-search methods. Answer EM/F1 and avg_search_count should be interpreted together.

| Method | Total | EM | F1 | Avg Search | Search Mismatch | Invalid Search Format | Answer Missing | Avg Answer Reward | Avg Final Reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | 500 | 0.098 | 0.169522 | 0 | 0 | 0 | 4 | 0.169522 | 0.161522 |
| always_retrieve | 500 | 0.03 | 0.187362 | 1 | 0 | 0 | 4 | 0.187362 | 0.379362 |
| rule_based | 500 | 0.096 | 0.175877 | 0.264 | 6 | 0 | 4 | 0.175877 | 0.193277 |
| prompt_self_routing | 500 | 0.022 | 0.159715 | 1.204 | 470 | 0 | 6 | 0.159715 | -0.053485 |
