from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from comparison_config import EnvironmentConfig


@dataclass
class EpisodeArtifacts:
    path_xy: np.ndarray
    segment_paths: List[np.ndarray]
    biases: List[Tuple[float, float, float, float]]
    final_distance: float
    success: bool


class AnalyticalPotentialEnv:
    def __init__(self, cfg: EnvironmentConfig):
        self.cfg = cfg
        self.action_tuples = [
            (float(a), float(w), float(o))
            for a in cfg.amp_bins
            for w in cfg.width_bins
            for o in cfg.offset_bins
        ]
        self.distance_history: deque[float] = deque(maxlen=10)
        self.episode_xy_segments: List[np.ndarray] = []
        self.all_biases_in_episode: List[Tuple[float, float, float, float]] = []
        self.milestones_reached: set[float] = set()
        self.stability_count = 0
        self.zone_confinement_active = False
        self.current_position = np.array([cfg.start_x, cfg.start_y], dtype=float)
        self.dist_to_target = self._dist_to_target(self.current_position)

    def copy_with_bias(self, enable_bias: bool) -> "AnalyticalPotentialEnv":
        new_cfg = EnvironmentConfig(**{**self.cfg.__dict__, "enable_bias": enable_bias})
        return AnalyticalPotentialEnv(new_cfg)

    def reset(
        self,
        start_position: Optional[Sequence[float]] = None,
        add_noise: bool = True,
        carry_state: bool = False,
    ) -> np.ndarray:
        if start_position is not None:
            self.current_position = np.asarray(start_position, dtype=float).copy()
        elif not carry_state or self._dist_to_target(self.current_position) <= self.cfg.target_radius:
            self.current_position = np.array([self.cfg.start_x, self.cfg.start_y], dtype=float)
            if add_noise and self.cfg.start_noise_std > 0:
                self.current_position += np.random.normal(0.0, self.cfg.start_noise_std, size=2)

        self._clip_to_domain()
        self.dist_to_target = self._dist_to_target(self.current_position)
        self.distance_history.clear()
        self.distance_history.append(self.dist_to_target)
        self.episode_xy_segments = []
        self.all_biases_in_episode = []
        self.stability_count = 0
        self.zone_confinement_active = self._in_target_zone(self.current_position)
        if not self.cfg.persist_locks_across_episodes:
            self.milestones_reached.clear()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        self.distance_history.append(self.dist_to_target)
        if len(self.distance_history) >= 3:
            recent_trend = (self.distance_history[-1] - self.distance_history[-3]) / 2.0
        else:
            recent_trend = 0.0
        if len(self.distance_history) >= 5:
            stability = 1.0 / (1.0 + np.std(list(self.distance_history)[-5:]))
        else:
            stability = 0.5
        start_dist = np.hypot(self.cfg.start_x - self.cfg.target_x, self.cfg.start_y - self.cfg.target_y)
        overall = 1.0 - (self.dist_to_target / (start_dist + 1e-12))
        state = np.array(
            [
                self.dist_to_target / (2.0 * np.pi),
                float(np.clip(overall, 0.0, 1.0)),
                recent_trend / 0.1,
                float(self.dist_to_target <= self.cfg.target_radius),
                stability,
                float(len(self.all_biases_in_episode) / max(1, self.cfg.max_biases)),
                self.current_position[0] / (2.0 * np.pi),
                self.current_position[1] / (2.0 * np.pi),
            ],
            dtype=np.float32,
        )
        return state

    @staticmethod
    def _multiwell_params():
        amp_kj = 6.0 * 4.184
        a_i = np.array([0.9, 0.3, 0.5, 1.0, 0.2, 0.4, 0.9, 0.9, 0.9], dtype=float) * amp_kj
        x0_i = np.array([1.12, 1.0, 3.0, 4.15, 4.0, 5.27, 5.5, 6.0, 1.0], dtype=float)
        y0_i = np.array([1.34, 2.25, 2.31, 3.62, 5.0, 4.14, 4.5, 1.52, 5.0], dtype=float)
        sx_i = np.array([0.5, 0.3, 0.4, 2.0, 0.9, 1.0, 0.3, 0.5, 0.5], dtype=float)
        sy_i = np.array([0.5, 0.3, 1.0, 0.8, 0.2, 0.3, 1.0, 0.6, 0.7], dtype=float)

        a_j = np.array([0.3], dtype=float) * amp_kj
        x0_j = np.array([np.pi], dtype=float)
        y0_j = np.array([np.pi], dtype=float)
        sx_j = np.array([3.0], dtype=float)
        sy_j = np.array([0.3], dtype=float)
        return amp_kj, a_i, x0_i, y0_i, sx_i, sy_i, a_j, x0_j, y0_j, sx_j, sy_j

    def background_potential(self, pos: Sequence[float]) -> float:
        x, y = float(pos[0]), float(pos[1])
        amp_kj, a_i, x0_i, y0_i, sx_i, sy_i, a_j, x0_j, y0_j, sx_j, sy_j = self._multiwell_params()
        u = float(amp_kj)
        dx = x - x0_i
        dy = y - y0_i
        arg = (dx * dx) / (2.0 * sx_i * sx_i) + (dy * dy) / (2.0 * sy_i * sy_i)
        u += float(-np.sum(a_i * np.exp(-arg)))
        dxj = x - x0_j
        dyj = y - y0_j
        argj = (dxj * dxj) / (2.0 * sx_j * sx_j) + (dyj * dyj) / (2.0 * sy_j * sy_j)
        u += float(np.sum(a_j * np.exp(-argj)))
        min_v = self.cfg.domain_min
        max_v = self.cfg.domain_max
        if x < min_v:
            u += self.cfg.wall_k * (min_v - x) ** 2
        elif x > max_v:
            u += self.cfg.wall_k * (x - max_v) ** 2
        if y < min_v:
            u += self.cfg.wall_k * (min_v - y) ** 2
        elif y > max_v:
            u += self.cfg.wall_k * (y - max_v) ** 2
        return float(u)

    def background_potential_grid(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        amp_kj, a_i, x0_i, y0_i, sx_i, sy_i, a_j, x0_j, y0_j, sx_j, sy_j = self._multiwell_params()
        z = np.full_like(x_grid, float(amp_kj), dtype=float)
        dx = x_grid[..., None] - x0_i
        dy = y_grid[..., None] - y0_i
        arg = (dx * dx) / (2.0 * sx_i * sx_i) + (dy * dy) / (2.0 * sy_i * sy_i)
        z += -np.sum(a_i * np.exp(-arg), axis=-1)
        dxj = x_grid[..., None] - x0_j
        dyj = y_grid[..., None] - y0_j
        argj = (dxj * dxj) / (2.0 * sx_j * sx_j) + (dyj * dyj) / (2.0 * sy_j * sy_j)
        z += np.sum(a_j * np.exp(-argj), axis=-1)
        return z

    def potential_force(self, pos: Sequence[float]) -> np.ndarray:
        x, y = float(pos[0]), float(pos[1])
        _, a_i, x0_i, y0_i, sx_i, sy_i, a_j, x0_j, y0_j, sx_j, sy_j = self._multiwell_params()
        dx = x - x0_i
        dy = y - y0_i
        arg = (dx * dx) / (2.0 * sx_i * sx_i) + (dy * dy) / (2.0 * sy_i * sy_i)
        exp_term = np.exp(-arg)
        fx = float(-np.sum(a_i * exp_term * dx / (sx_i * sx_i)))
        fy = float(-np.sum(a_i * exp_term * dy / (sy_i * sy_i)))

        dxj = x - x0_j
        dyj = y - y0_j
        argj = (dxj * dxj) / (2.0 * sx_j * sx_j) + (dyj * dyj) / (2.0 * sy_j * sy_j)
        exp_barrier = np.exp(-argj)
        fx += float(np.sum(a_j * exp_barrier * dxj / (sx_j * sx_j)))
        fy += float(np.sum(a_j * exp_barrier * dyj / (sy_j * sy_j)))

        min_v = self.cfg.domain_min
        max_v = self.cfg.domain_max
        if x < min_v:
            fx += 2.0 * self.cfg.wall_k * (min_v - x)
        elif x > max_v:
            fx -= 2.0 * self.cfg.wall_k * (x - max_v)
        if y < min_v:
            fy += 2.0 * self.cfg.wall_k * (min_v - y)
        elif y > max_v:
            fy -= 2.0 * self.cfg.wall_k * (y - max_v)
        return np.array([fx, fy], dtype=float)

    def bias_potential(self, pos: Sequence[float], biases: Optional[Iterable[Tuple[float, float, float, float]]] = None) -> float:
        if not self.cfg.enable_bias:
            return 0.0
        if biases is None:
            biases = self.all_biases_in_episode
        x, y = float(pos[0]), float(pos[1])
        energy = 0.0
        for amp, x0, y0, sigma in biases:
            if amp == 0.0:
                continue
            sigma = max(float(sigma), 1e-6)
            dx = x - float(x0)
            dy = y - float(y0)
            energy += float(amp) * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        return float(energy)

    def bias_potential_grid(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        biases: Optional[Iterable[Tuple[float, float, float, float]]] = None,
    ) -> np.ndarray:
        if not self.cfg.enable_bias:
            return np.zeros_like(x_grid, dtype=float)
        if biases is None:
            biases = self.all_biases_in_episode
        bias_grid = np.zeros_like(x_grid, dtype=float)
        for amp, x0, y0, sigma in biases:
            if amp == 0.0:
                continue
            sigma = max(float(sigma), 1e-6)
            dx = x_grid - float(x0)
            dy = y_grid - float(y0)
            bias_grid += float(amp) * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        return bias_grid

    def bias_force(
        self,
        pos: Sequence[float],
        biases: Optional[Iterable[Tuple[float, float, float, float]]] = None,
        fmax: float = 200.0,
    ) -> np.ndarray:
        if not self.cfg.enable_bias:
            return np.zeros(2, dtype=float)
        if biases is None:
            biases = self.all_biases_in_episode
        x, y = float(pos[0]), float(pos[1])
        total_force = np.zeros(2, dtype=float)
        for amp, x0, y0, sigma in biases:
            if amp == 0.0:
                continue
            sigma = max(float(sigma), 1e-6)
            dx = x - float(x0)
            dy = y - float(y0)
            exp_term = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            bias = float(amp) * exp_term
            coef = bias / (sigma * sigma)
            total_force[0] += coef * dx
            total_force[1] += coef * dy
        norm = float(np.linalg.norm(total_force))
        if norm > fmax:
            total_force *= fmax / (norm + 1e-12)
        return total_force

    def total_potential(
        self,
        pos: Sequence[float],
        biases: Optional[Iterable[Tuple[float, float, float, float]]] = None,
    ) -> float:
        return self.background_potential(pos) + self.bias_potential(pos, biases)

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        action_index = int(action_index)
        prev_dist = self.dist_to_target
        if 0 <= action_index < len(self.action_tuples):
            amp, width, offset = self.action_tuples[action_index]
        else:
            amp, width, offset = 0.0, 1.0, 0.0
        if prev_dist <= self.cfg.target_radius or not self.cfg.enable_bias:
            amp = 0.0

        if self.cfg.enable_bias and amp != 0.0:
            x0, y0 = self._bias_center(offset)
            self.all_biases_in_episode.append((float(amp), x0, y0, float(width)))
            if len(self.all_biases_in_episode) > self.cfg.max_biases:
                self.all_biases_in_episode = self.all_biases_in_episode[-self.cfg.max_biases :]

        d = self.cfg.temperature / self.cfg.friction
        sigma_noise = np.sqrt(2.0 * d * self.cfg.dt)
        traj_xy: List[np.ndarray] = []
        done = False
        for _ in range(self.cfg.sim_steps):
            force = self.potential_force(self.current_position)
            force += self.bias_force(self.current_position, self.all_biases_in_episode)
            if self.cfg.enable_milestone_locks and self.milestones_reached:
                lock_dist = min(self.milestones_reached)
                force += self._lock_force(self.current_position, lock_dist)
            if self.cfg.zone_confinement:
                force += self._zone_force(self.current_position)
            disp = (force / self.cfg.friction) * self.cfg.dt
            disp += np.random.normal(0.0, 1.0, 2) * sigma_noise
            self.current_position += disp
            self._clip_to_domain()
            self.dist_to_target = self._dist_to_target(self.current_position)
            traj_xy.append(self.current_position.copy())
            if self.dist_to_target <= self.cfg.target_radius:
                self.zone_confinement_active = True
                done = True
                break
        traj_xy_np = np.asarray(traj_xy, dtype=float)
        self.episode_xy_segments.append(traj_xy_np)

        progress = prev_dist - self.dist_to_target
        reward = self.cfg.progress_reward * progress
        if progress < 0:
            reward += self.cfg.backtrack_penalty
        reward += self.cfg.step_penalty
        if self.cfg.enable_bias:
            reward -= self.cfg.bias_penalty * abs(float(amp))
        for milestone in self.cfg.distance_increments:
            if self.dist_to_target <= float(milestone) and milestone not in self.milestones_reached:
                reward += self.cfg.milestone_reward
                self.milestones_reached.add(float(milestone))
        if self.dist_to_target <= self.cfg.phase2_tol:
            self.stability_count += 1
            if self.stability_count == self.cfg.stability_steps:
                reward += self.cfg.consistency_bonus
        else:
            self.stability_count = 0
        if done:
            reward += self.cfg.terminal_bonus
        return self.get_state(), float(reward), bool(done), traj_xy_np

    def episode_artifacts(self) -> EpisodeArtifacts:
        if self.episode_xy_segments:
            path_xy = np.concatenate(self.episode_xy_segments, axis=0)
        else:
            path_xy = np.empty((0, 2), dtype=float)
        return EpisodeArtifacts(
            path_xy=path_xy,
            segment_paths=list(self.episode_xy_segments),
            biases=list(self.all_biases_in_episode),
            final_distance=float(self.dist_to_target),
            success=bool(self.dist_to_target <= self.cfg.target_radius),
        )

    def _bias_center(self, offset: float) -> Tuple[float, float]:
        if self.cfg.bias_placement_mode == "current_position":
            return float(self.current_position[0]), float(self.current_position[1])
        if self.cfg.bias_placement_mode != "away_from_target":
            raise ValueError(f"Unknown bias placement mode: {self.cfg.bias_placement_mode}")
        target = np.array([self.cfg.target_x, self.cfg.target_y], dtype=float)
        vec_away = self.current_position - target
        norm = float(np.linalg.norm(vec_away))
        if norm < 1e-12:
            theta = np.random.uniform(0, 2.0 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        else:
            direction = vec_away / norm
        center_xy = self.current_position + float(offset) * direction
        return float(center_xy[0]), float(center_xy[1])

    def _dist_to_target(self, pos: Sequence[float]) -> float:
        dx = float(pos[0]) - self.cfg.target_x
        dy = float(pos[1]) - self.cfg.target_y
        return float(np.hypot(dx, dy))

    def _clip_to_domain(self) -> None:
        min_v = self.cfg.domain_min
        max_v = self.cfg.domain_max
        for idx in (0, 1):
            value = float(self.current_position[idx])
            while value < min_v or value > max_v:
                if value < min_v:
                    value = min_v + (min_v - value)
                elif value > max_v:
                    value = max_v - (value - max_v)
            self.current_position[idx] = float(np.clip(value, min_v, max_v))

    def _lock_force(self, pos: Sequence[float], lock_dist: float) -> np.ndarray:
        dist = self._dist_to_target(pos)
        if dist <= lock_dist + self.cfg.lock_margin:
            return np.zeros(2, dtype=float)
        target = np.array([self.cfg.target_x, self.cfg.target_y], dtype=float)
        vec = target - np.asarray(pos, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.zeros(2, dtype=float)
        return (vec / norm) * self.cfg.backstop_k * (dist - (lock_dist + self.cfg.lock_margin))

    def _zone_force(self, pos: Sequence[float]) -> np.ndarray:
        if not self.zone_confinement_active:
            return np.zeros(2, dtype=float)
        dist = self._dist_to_target(pos)
        max_allowed = self.cfg.target_radius + self.cfg.zone_margin_high
        if dist <= max_allowed:
            return np.zeros(2, dtype=float)
        target = np.array([self.cfg.target_x, self.cfg.target_y], dtype=float)
        vec = target - np.asarray(pos, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.zeros(2, dtype=float)
        return (vec / norm) * self.cfg.zone_k * (dist - max_allowed)

    def _in_target_zone(self, pos: Sequence[float]) -> bool:
        return self._dist_to_target(pos) <= self.cfg.target_radius
