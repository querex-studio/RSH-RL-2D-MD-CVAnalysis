# Model Config Overview

| model | controller | encoder | bias_enabled | budget_env_steps | warmup_env_steps | train_env_steps | approx_episodes_or_rounds | max_actions_per_episode | eval_every | eval_episodes | lag_or_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_biased | PPO | identity | True | 40000 | 0 | 40000 | 100 | 40 | 20 | 8 | - |
| adaptive_cvgen_like_2d | Adaptive-CVgen-like | TICA clustering | False | 40000 | 0 | 40000 | 50 | 10 | - | - | tica_lag=5, alpha_candidates=[0.0, 0.1, 0.25, 0.5, 1.0, 1.5] |
| ppo_tica_2d | PPO | TICA | True | 40000 | 11520 | 28480 | 71 | 40 | 20 | 8 | lag=5, comps=4 |
| ppo_vampnet_2d | PPO | VAMPNet | True | 40000 | 11520 | 28480 | 71 | 40 | 20 | 8 | lag=5, comps=4, epochs=60 |
