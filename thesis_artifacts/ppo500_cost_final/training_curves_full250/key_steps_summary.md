# Key PPO Steps Summary

| step | reward_decomposition/total_token_score_sum_mean | reward_decomposition/search_reward_sum_mean | reward_decomposition/answer_reward_sum_mean | reward_decomposition/invalid_search_count_mean | ppo_chain/token_level_scores_mean | ppo_chain/token_level_rewards_mean | actor/pg_loss | critic/vf_loss | env/number_of_actions/mean | env/number_of_valid_search | env/finish_ratio | response_length/mean | val/test_score/nq | val/test_score/hotpotqa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 |  |  |  |  |  |  |  |  |  |  |  |  | 0.020000 | 0.020000 |
| 50 | -0.300000 | -0.300000 | 0.000000 | 0.000000 | -0.001000 | -0.001000 | 0.075000 | 0.026000 | 3.000000 | 1.000000 | 0.000000 | 467.500000 | 0.000000 | 0.020000 |
| 100 | -0.550000 | -0.050000 | 0.000000 | 0.000000 | -0.002000 | -0.002000 | -0.001000 | 0.272000 | 2.000000 | 0.000000 | 1.000000 | 241.500000 | 0.041000 | 0.078000 |
| 150 | -0.125000 | -0.125000 | 0.500000 | 0.000000 | -0.000000 | -0.000000 | -0.409000 | 1.331000 | 2.500000 | 0.500000 | 1.000000 | 365.500000 | 0.020000 | 0.059000 |
| 200 | -0.050000 | -0.050000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | -0.075000 | 0.001000 | 2.000000 | 0.000000 | 1.000000 | 201.000000 | 0.000000 | 0.059000 |
| 250 |  |  |  |  |  |  |  |  |  |  |  |  | 0.020000 | 0.059000 |
