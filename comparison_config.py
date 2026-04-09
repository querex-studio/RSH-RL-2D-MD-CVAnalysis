from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class EnvironmentConfig:
    start_x: float = 5.0
    start_y: float = 4.0
    target_x: float = 1.0
    target_y: float = 1.5
    target_radius: float = 0.2

    state_size: int = 8
    amp_bins: Sequence[float] = field(default_factory=lambda: [0.0, 2.0, 4.0, 6.0, 8.0])
    width_bins: Sequence[float] = field(default_factory=lambda: [0.2, 0.5, 0.8, 1.1])
    offset_bins: Sequence[float] = field(default_factory=lambda: [0.8, 1.6, 2.4, 3.2])

    bias_placement_mode: str = "current_position"
    enable_bias: bool = True
    max_biases: int = 100
    min_amp: float = 0.0
    max_amp: float = 30.0
    min_width: float = 0.2
    max_width: float = 2.5
    in_zone_max_amp: float = 1e9

    enable_milestone_locks: bool = False
    persist_locks_across_episodes: bool = True
    free_exploration_at_zone: bool = False
    zone_confinement: bool = True
    zone_k: float = 1000.0
    zone_margin_high: float = 0.1
    lock_margin: float = 0.15
    backstop_k: float = 1000.0
    distance_increments: Sequence[float] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    sim_steps: int = 10
    dt: float = 0.01
    temperature: float = 1.0
    friction: float = 1.0

    progress_reward: float = 10.0
    milestone_reward: float = 100.0
    backtrack_penalty: float = -1.0
    step_penalty: float = -0.1
    bias_penalty: float = 0.01
    phase2_tol: float = 0.1
    stability_steps: int = 10
    consistency_bonus: float = 20.0
    terminal_bonus: float = 200.0

    wall_k: float = 5000.0
    start_noise_std: float = 0.05
    plot_grid_size: int = 220
    trajectory_dpi: int = 160

    @property
    def action_size(self) -> int:
        return len(self.amp_bins) * len(self.width_bins) * len(self.offset_bins)

    @property
    def domain_min(self) -> float:
        return 0.0

    @property
    def domain_max(self) -> float:
        return 2.0 * np.pi


@dataclass
class PPOConfig:
    n_steps: int = 16
    batch_size: int = 16
    n_epochs: int = 6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    target_kl: float = 0.03
    scheduler_step: int = 100
    scheduler_gamma: float = 0.95
    hidden_sizes: Sequence[int] = field(default_factory=lambda: [256, 128, 64])

    max_actions_per_episode: int = 40
    eval_every: int = 20
    n_eval_episodes: int = 8
    moving_avg_window: int = 20
    eval_greedy: bool = True

    exploration_noise: float = 0.05
    min_exploration_noise: float = 0.005
    exploration_decay: float = 0.99


@dataclass
class EncoderFitConfig:
    warmup_episodes: int = 48
    warmup_actions_per_episode: int = 24
    lagtime: int = 5
    n_components: int = 4
    feature_basis: str = "state"


@dataclass
class VAMPNetConfig:
    hidden_sizes: Sequence[int] = field(default_factory=lambda: [64, 64])
    n_components: int = 4
    lagtime: int = 5
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    score_eps: float = 1e-4


@dataclass
class AdaptiveCVgenConfig:
    replicas: int = 8
    actions_per_segment: int = 10
    random_candidates_per_cluster: int = 4
    n_clusters: int = 18
    candidate_stride: int = 2
    anchor_grid_size: int = 6
    anchor_sigma: float = 0.55
    alpha_candidates: Sequence[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0, 1.5])
    alpha_entropy_coef: float = 0.35
    alpha_dispersion_penalty: float = 0.05
    tica_lagtime: int = 5
    tica_components: int = 2
    min_history_for_tica: int = 30
    moving_avg_window: int = 10


@dataclass
class ComparisonConfig:
    seed: int = 42
    total_env_steps_budget: int = 40_000
    output_root_name: str = "comparison_outputs"

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    encoder_fit: EncoderFitConfig = field(default_factory=EncoderFitConfig)
    vampnet: VAMPNetConfig = field(default_factory=VAMPNetConfig)
    adaptive: AdaptiveCVgenConfig = field(default_factory=AdaptiveCVgenConfig)

    def output_root(self) -> Path:
        return Path(__file__).resolve().parent / self.output_root_name

    def model_dir(self, model_name: str) -> Path:
        return self.output_root() / model_name
