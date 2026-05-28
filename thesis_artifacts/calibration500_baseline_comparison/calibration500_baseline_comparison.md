# Calibration-500 Baseline Comparison

> Warning: avg_final_reward may not be directly comparable across baselines if search_decision_reward gives fixed reward to forced-search methods. Answer EM/F1 and avg_search_count should be interpreted together.

| Method | Total | EM | F1 | Avg Search | Search Mismatch | Invalid Search Format | Answer Missing | Avg Answer Reward | Avg Final Reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | 500 | 0.072 | 0.147648 | 0 | 0 | 0 | 2 | 0.147648 | 0.143648 |
| always_retrieve | 500 | 0.012 | 0.138025 | 1 | 0 | 0 | 4 | 0.138025 | 0.330025 |
| rule_based | 500 | 0.06 | 0.141974 | 0.212 | 7 | 0 | 3 | 0.141974 | 0.168574 |
| prompt_self_routing | 500 | 0.016 | 0.122445 | 1.102 | 470 | 0 | 5 | 0.122445 | -0.083655 |
