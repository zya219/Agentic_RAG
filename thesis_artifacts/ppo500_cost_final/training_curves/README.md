# strict_query_token_cost PPO-500 Training Curves

Source log: `/root/autodl-tmp/ARAG/experiment_archive/ppo500_logs/strict_query_token_cost_ppo500_qwen05b_t2_r128_0511_2201.log`
CSV: `/root/autodl-tmp/ARAG/experiment_archive/ppo500_cost_final/training_curves/strict_query_token_cost_ppo500_training_metrics.csv`

## Available plots
- `reward_decomposition/total_token_score_sum_mean` → `reward_decomposition__total_token_score_sum_mean.png`
- `reward_decomposition/search_reward_sum_mean` → `reward_decomposition__search_reward_sum_mean.png`
- `reward_decomposition/answer_reward_sum_mean` → `reward_decomposition__answer_reward_sum_mean.png`
- `reward_decomposition/invalid_search_count_mean` → `reward_decomposition__invalid_search_count_mean.png`
- `ppo_chain/token_level_scores_mean` → `ppo_chain__token_level_scores_mean.png`
- `ppo_chain/token_level_rewards_mean` → `ppo_chain__token_level_rewards_mean.png`
- `ppo_chain/advantages_mean` → `ppo_chain__advantages_mean.png`
- `actor/pg_loss` → `actor__pg_loss.png`
- `actor/ppo_kl` → `actor__ppo_kl.png`
- `actor/entropy_loss` → `actor__entropy_loss.png`
- `critic/vf_loss` → `critic__vf_loss.png`
- `critic/vf_explained_var` → `critic__vf_explained_var.png`
- `critic/kl` → `critic__kl.png`
- `env/number_of_actions/mean` → `env__number_of_actions__mean.png`
- `env/number_of_valid_search` → `env__number_of_valid_search.png`
- `env/finish_ratio` → `env__finish_ratio.png`
- `response_length/mean` → `response_length__mean.png`
- `response_length/clip_ratio` → `response_length__clip_ratio.png`
- `val/test_score/nq` → `val__test_score__nq.png`
- `val/test_score/hotpotqa` → `val__test_score__hotpotqa.png`
