# Final Config And Results

| model | controller | encoder | budget_env_steps | warmup_env_steps | train_env_steps | approx_episodes_or_rounds | success_rate | best_final_distance | n_successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_biased | PPO | identity | 40000 | 0 | 40000 | 100 | 0.13 | 0.02311843864437354 | 13 |
| adaptive_cvgen_like_2d | Adaptive-CVgen-like | TICA clustering | 40000 | 0 | 40000 | 50 | 0.04 | 0.019241596725958782 | 2 |
| ppo_tica_2d | PPO | TICA | 40000 | 11520 | 28480 | 71 | 0.14084507042253522 | 0.05756029550123656 | 10 |
| ppo_vampnet_2d | PPO | VAMPNet | 40000 | 11520 | 28480 | 71 | 0.16901408450704225 | 0.06583178350378535 | 12 |
