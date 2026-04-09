from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from comparison_config import AdaptiveCVgenConfig, EncoderFitConfig, EnvironmentConfig
from comparison_encoders import TICAEncoder
from comparison_env import AnalyticalPotentialEnv


@dataclass
class SeedPath:
    position: np.ndarray
    path_xy: np.ndarray
    score: float
    round_index: int


@dataclass
class CandidateFrame:
    position: np.ndarray
    full_path: np.ndarray
    reward_score: float
    round_index: int
    success: bool


class AdaptiveCVgen2D:
    def __init__(self, env_cfg: EnvironmentConfig, cfg: AdaptiveCVgenConfig, seed: int):
        self.env_cfg = EnvironmentConfig(**{**env_cfg.__dict__, "enable_bias": False})
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.env = AnalyticalPotentialEnv(self.env_cfg)
        self.anchor_centers = self._build_anchor_centers()
        self.gamma = self._build_prior_rewards()
        self.history_positions: List[np.ndarray] = []
        self.successful_paths: List[CandidateFrame] = []
        self.history: List[Dict[str, float]] = []
        self.frontier_archive: List[CandidateFrame] = []

    def run(self, total_env_steps_budget: int) -> Dict[str, object]:
        steps_per_round = self.cfg.replicas * self.cfg.actions_per_segment * self.env_cfg.sim_steps
        n_rounds = max(1, total_env_steps_budget // max(1, steps_per_round))
        start = np.array([self.env_cfg.start_x, self.env_cfg.start_y], dtype=float)
        current_seeds = [SeedPath(position=start.copy(), path_xy=start[None, :], score=0.0, round_index=0)]
        for round_idx in range(1, n_rounds + 1):
            trajectories = self._sample_round(current_seeds, round_idx)
            current_candidates = self._candidate_frames_from_round(trajectories, round_idx)
            if not current_candidates:
                break
            current_positions = np.array([cand.position for cand in current_candidates], dtype=np.float32)
            self.history_positions.extend([pos.copy() for pos in current_positions])
            theta_hist = self._theta(np.array(self.history_positions, dtype=np.float32))
            alpha, weights = self._select_alpha(theta_hist)
            occupancy = np.sum(theta_hist, axis=0)
            occupancy_norm = occupancy / (np.max(occupancy) + 1e-8)
            candidate_scores = self._theta(current_positions) @ weights
            for cand, score in zip(current_candidates, candidate_scores):
                local_theta = self._theta(cand.position[None, :])[0]
                local_occupancy = float(np.sum(local_theta * occupancy_norm) / (np.sum(local_theta) + 1e-8))
                target_bonus = 1.0 / (self.env._dist_to_target(cand.position) + 0.2)
                novelty_bonus = 1.0 - local_occupancy
                cand.reward_score = float(0.35 * score + 0.55 * target_bonus + 0.10 * novelty_bonus)
                if cand.success:
                    self.successful_paths.append(cand)
            self._update_frontier_archive(current_candidates)
            selected = self._select_next_seeds(current_candidates, round_idx)
            best_distance = min(self.env._dist_to_target(cand.position) for cand in current_candidates)
            success_count = float(sum(1 for cand in current_candidates if cand.success))
            coverage = self._coverage_ratio(np.array(self.history_positions, dtype=np.float32))
            self.history.append(
                {
                    "round": int(round_idx),
                    "total_env_steps": int(round_idx * steps_per_round),
                    "best_distance": float(best_distance),
                    "mean_distance": float(np.mean([self.env._dist_to_target(c.position) for c in current_candidates])),
                    "successes_this_round": success_count,
                    "cumulative_successes": float(len(self.successful_paths)),
                    "alpha": float(alpha),
                    "coverage_ratio": float(coverage),
                }
            )
            if selected:
                current_seeds = selected
            else:
                best = max(current_candidates, key=lambda item: item.reward_score)
                current_seeds = [SeedPath(best.position.copy(), best.full_path.copy(), best.reward_score, round_idx)]
        return {
            "history": self.history,
            "successful_paths": self._best_successful_paths(),
            "final_alpha": float(self.history[-1]["alpha"]) if self.history else 0.0,
        }

    def _sample_round(self, seeds: Sequence[SeedPath], round_index: int):
        replicas = []
        for replica_idx in range(self.cfg.replicas):
            seed = seeds[min(replica_idx, len(seeds) - 1)]
            state = self.env.reset(start_position=seed.position, add_noise=False)
            segment_points = [seed.position.copy()]
            done = False
            for _ in range(self.cfg.actions_per_segment):
                state, _, done, segment_xy = self.env.step(0)
                if len(segment_xy) > 0:
                    segment_points.extend(segment_xy)
                if done:
                    break
            replicas.append(
                {
                    "seed": seed,
                    "segment_xy": np.asarray(segment_points, dtype=float),
                    "done": bool(done),
                    "round_index": round_index,
                }
            )
        return replicas

    def _candidate_frames_from_round(self, trajectories, round_index: int) -> List[CandidateFrame]:
        current_candidates: List[CandidateFrame] = []
        for traj in trajectories:
            seed: SeedPath = traj["seed"]
            segment_xy = np.asarray(traj["segment_xy"], dtype=float)
            if len(segment_xy) <= 1:
                continue
            indices = list(range(1, len(segment_xy), self.cfg.candidate_stride))
            if (len(segment_xy) - 1) not in indices:
                indices.append(len(segment_xy) - 1)
            for frame_idx in indices:
                prefix = segment_xy[1 : frame_idx + 1]
                full_path = np.vstack([seed.path_xy, prefix])
                position = full_path[-1]
                success = bool(self.env._dist_to_target(position) <= self.env_cfg.target_radius)
                current_candidates.append(
                    CandidateFrame(
                        position=position.copy(),
                        full_path=full_path,
                        reward_score=0.0,
                        round_index=round_index,
                        success=success,
                    )
                )
        return current_candidates

    def _build_anchor_centers(self) -> np.ndarray:
        xs = np.linspace(self.env_cfg.domain_min + 0.2, self.env_cfg.domain_max - 0.2, self.cfg.anchor_grid_size)
        ys = np.linspace(self.env_cfg.domain_min + 0.2, self.env_cfg.domain_max - 0.2, self.cfg.anchor_grid_size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

    def _build_prior_rewards(self) -> np.ndarray:
        target = np.array([self.env_cfg.target_x, self.env_cfg.target_y], dtype=np.float32)
        dist = np.linalg.norm(self.anchor_centers - target[None, :], axis=1)
        gamma = 1.0 / (dist + 0.25)
        gamma /= np.sum(gamma)
        return gamma.astype(np.float32)

    def _theta(self, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float32)
        diff = pos[:, None, :] - self.anchor_centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        theta = np.exp(-dist2 / (2.0 * self.cfg.anchor_sigma * self.cfg.anchor_sigma))
        return theta.astype(np.float32)

    def _select_alpha(self, theta_hist: np.ndarray):
        occupancy = np.sum(theta_hist, axis=0)
        occupancy_norm = occupancy / (np.max(occupancy) + 1e-8)
        delta = -occupancy_norm
        candidate_scores = theta_hist @ self.gamma
        best_alpha = 0.0
        best_weights = self.gamma.copy()
        best_objective = -np.inf
        for alpha in self.cfg.alpha_candidates:
            weights = self.gamma * np.exp(alpha * delta)
            weights = weights / (np.sum(weights) + 1e-8)
            scores = theta_hist @ weights
            top_k = max(5, int(0.1 * len(scores)))
            exploit = float(np.mean(np.sort(scores)[-top_k:]))
            entropy = float(-(weights * np.log(weights + 1e-8)).sum() / np.log(len(weights)))
            dispersion = float(np.std(weights))
            objective = exploit + self.cfg.alpha_entropy_coef * entropy - self.cfg.alpha_dispersion_penalty * dispersion
            objective += 0.05 * float(np.mean(scores >= np.median(candidate_scores)))
            if objective > best_objective:
                best_objective = objective
                best_alpha = float(alpha)
                best_weights = weights.astype(np.float32)
        return best_alpha, best_weights

    def _select_next_seeds(self, candidates: List[CandidateFrame], round_index: int) -> List[SeedPath]:
        candidate_pool = list(candidates) + list(self.frontier_archive)
        positions = np.array([cand.position for cand in candidate_pool], dtype=np.float32)
        projections = self._project_history_and_candidates(positions)
        n_clusters = min(self.cfg.n_clusters, len(candidate_pool))
        if n_clusters <= 1:
            ordered = sorted(candidate_pool, key=lambda item: (self.env._dist_to_target(item.position), -item.reward_score))
            return [
                SeedPath(item.position.copy(), item.full_path.copy(), item.reward_score, round_index)
                for item in ordered[: self.cfg.replicas]
            ]
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, batch_size=1024, n_init="auto")
        labels = kmeans.fit_predict(projections)
        ordered_global = sorted(candidate_pool, key=lambda item: (self.env._dist_to_target(item.position), -item.reward_score))
        chosen: List[CandidateFrame] = ordered_global[: max(2, self.cfg.replicas // 2)]
        chosen_ids = {id(item) for item in chosen}
        for cluster_id in range(n_clusters):
            members = [candidate_pool[idx] for idx, label in enumerate(labels) if label == cluster_id]
            if not members:
                continue
            take = min(len(members), self.cfg.random_candidates_per_cluster)
            sampled = list(self.rng.choice(members, size=take, replace=False))
            sampled.sort(key=lambda item: (self.env._dist_to_target(item.position), -item.reward_score))
            best = sampled[0]
            if id(best) not in chosen_ids:
                chosen.append(best)
                chosen_ids.add(id(best))
        chosen.sort(key=lambda item: item.reward_score, reverse=True)
        return [
            SeedPath(item.position.copy(), item.full_path.copy(), item.reward_score, round_index)
            for item in chosen[: self.cfg.replicas]
        ]

    def _project_history_and_candidates(self, current_positions: np.ndarray) -> np.ndarray:
        history = np.array(self.history_positions, dtype=np.float32)
        combined = np.vstack([history, current_positions])
        if len(combined) < max(self.cfg.min_history_for_tica, self.cfg.tica_lagtime + 3):
            return combined[:, :2]
        fit_cfg = EncoderFitConfig(
            warmup_episodes=0,
            warmup_actions_per_episode=0,
            lagtime=self.cfg.tica_lagtime,
            n_components=self.cfg.tica_components,
            feature_basis="augmented",
        )
        encoder = TICAEncoder(fit_cfg)
        pseudo_states = self._positions_to_states(combined)
        encoder.fit(pseudo_states)
        projections = encoder.transform_batch(pseudo_states)
        return projections[-len(current_positions) :]

    def _positions_to_states(self, positions: np.ndarray) -> np.ndarray:
        states = []
        start_dist = np.hypot(self.env_cfg.start_x - self.env_cfg.target_x, self.env_cfg.start_y - self.env_cfg.target_y)
        for pos in np.asarray(positions, dtype=np.float32):
            dist = self.env._dist_to_target(pos)
            overall = 1.0 - dist / (start_dist + 1e-12)
            states.append(
                [
                    dist / (2.0 * np.pi),
                    np.clip(overall, 0.0, 1.0),
                    0.0,
                    float(dist <= self.env_cfg.target_radius),
                    0.5,
                    0.0,
                    pos[0] / (2.0 * np.pi),
                    pos[1] / (2.0 * np.pi),
                ]
            )
        return np.asarray(states, dtype=np.float32)

    def _coverage_ratio(self, positions: np.ndarray) -> float:
        if len(positions) == 0:
            return 0.0
        theta = self._theta(positions)
        occupied = np.sum(theta, axis=0) > 0.5
        return float(np.mean(occupied))

    def _best_successful_paths(self) -> List[CandidateFrame]:
        if not self.successful_paths:
            return []
        ordered = sorted(
            self.successful_paths,
            key=lambda item: (-item.reward_score, len(item.full_path)),
        )
        return ordered[:2]

    def _update_frontier_archive(self, candidates: List[CandidateFrame]) -> None:
        merged = list(self.frontier_archive) + list(candidates)
        merged.sort(key=lambda item: (self.env._dist_to_target(item.position), -item.reward_score, len(item.full_path)))
        deduped: List[CandidateFrame] = []
        for item in merged:
            if any(np.linalg.norm(item.position - other.position) < 0.05 for other in deduped):
                continue
            deduped.append(item)
            if len(deduped) >= 24:
                break
        self.frontier_archive = deduped
