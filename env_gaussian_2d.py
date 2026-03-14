import numpy as np
import config_gaussian as config
from collections import deque


class Gaussian2DEnvironment:
    """
    2D analytic environment with:
      - Fixed background PSP: multiwell (negative Gaussians) + barrier (positive Gaussian) + hard walls
      - Adaptive bias: sum of 2D Gaussian hills in (x,y)
      - Overdamped Langevin (Euler–Maruyama): x <- x + (F/gamma)dt + sqrt(2Ddt)N(0,1)
    """

    def __init__(self):
        self.state_size = config.STATE_SIZE
        self.action_size = config.ACTION_SIZE

        self.current_position = np.array([0.0, 0.0], dtype=float)
        self.dist_to_target = 0.0

        self.distance_history = deque(maxlen=10)
        self.episode_trajectory_segments = []   # store dist_to_target over MD steps
        self.episode_xy_segments = []           # store (x,y) over MD steps

        self.all_biases_in_episode = []
        self.milestones_reached = set()
        self.stability_count = 0
        self.zone_confinement_active = False

        # action map (A, width, offset)
        self.action_tuples = []
        for A in config.AMP_BINS:
            for W in config.WIDTH_BINS:
                for O in config.OFFSET_BINS:
                    self.action_tuples.append((float(A), float(W), float(O)))

        self._validate_action_bins()
        self.reset()

    # -----------------------
    # Reset / State
    # -----------------------
    def reset(self, seed_from_max_A=None, carry_state=False, episode_index=None):
        carry_state = bool(carry_state)
        if carry_state and self._dist_to_target(self.current_position) <= float(config.TARGET_RADIUS):
            carry_state = False

        # always start from START_X/START_Y unless you explicitly carry_state=True
        if not carry_state:
            self.current_position = np.array([float(config.START_X), float(config.START_Y)], dtype=float)
            self.current_position += np.random.normal(0, 0.05, size=2)

        self._clip_to_domain()

        self.dist_to_target = self._dist_to_target(self.current_position)
        self.zone_confinement_active = self._in_target_zone(self.current_position)
        self.distance_history.clear()
        self.distance_history.append(self.dist_to_target)

        self.all_biases_in_episode = []
        self.episode_trajectory_segments = []
        self.episode_xy_segments = []
        self.stability_count = 0
        if not config.PERSIST_LOCKS_ACROSS_EPISODES:
            self.milestones_reached.clear()

        return self.get_state()

    def get_state(self):
        """
        Keep STATE_SIZE=8 but make features target-centric (XY target),
        so PPO input shape stays unchanged.
        """
        self.distance_history.append(self.dist_to_target)

        # trend: decreasing dist is good (negative trend)
        if len(self.distance_history) >= 3:
            recent_trend = (self.distance_history[-1] - self.distance_history[-3]) / 2.0
        else:
            recent_trend = 0.0

        # stability from recent distance variance
        if len(self.distance_history) >= 5:
            stability = 1.0 / (1.0 + np.std(list(self.distance_history)[-5:]))
        else:
            stability = 0.5

        # normalized distance (scale by typical map size ~2π)
        dist_norm = self.dist_to_target / (2.0 * np.pi)

        in_zone = float(self.dist_to_target <= float(config.TARGET_RADIUS))

        # simple "overall progress" proxy: compare to start distance
        start_dist = np.sqrt((config.START_X - config.TARGET_X) ** 2 + (config.START_Y - config.TARGET_Y) ** 2)
        overall = 1.0 - (self.dist_to_target / (start_dist + 1e-12))
        overall = float(np.clip(overall, 0.0, 1.0))

        state = np.array(
            [
                dist_norm,          # 0
                overall,            # 1
                recent_trend / 0.1, # 2
                in_zone,            # 3
                stability,          # 4
                float(len(self.all_biases_in_episode) / max(1, config.MAX_BIASES)),  # 5
                self.current_position[0] / (2.0 * np.pi),  # 6
                self.current_position[1] / (2.0 * np.pi),  # 7
            ],
            dtype=np.float32,
        )
        return state

    # -----------------------
    # Background PSP (your OpenMM multiwell design)
    # -----------------------
    @staticmethod
    def _multiwell_params():
        amp_kj = 6.0 * 4.184  # kJ/mol
        A_i = np.array([0.9, 0.3, 0.5, 1.0, 0.2, 0.4, 0.9, 0.9, 0.9], dtype=float) * amp_kj
        x0_i = np.array([1.12, 1.0, 3.0, 4.15, 4.0, 5.27, 5.5, 6.0, 1.0], dtype=float)
        y0_i = np.array([1.34, 2.25, 2.31, 3.62, 5.0, 4.14, 4.5, 1.52, 5.0], dtype=float)
        sx_i = np.array([0.5, 0.3, 0.4, 2.0, 0.9, 1.0, 0.3, 0.5, 0.5], dtype=float)
        sy_i = np.array([0.5, 0.3, 1.0, 0.8, 0.2, 0.3, 1.0, 0.6, 0.7], dtype=float)

        A_j = np.array([0.3], dtype=float) * amp_kj
        x0_j = np.array([np.pi], dtype=float)
        y0_j = np.array([np.pi], dtype=float)
        sx_j = np.array([3.0], dtype=float)
        sy_j = np.array([0.3], dtype=float)

        return amp_kj, A_i, x0_i, y0_i, sx_i, sy_i, A_j, x0_j, y0_j, sx_j, sy_j

    def background_potential(self, pos):
        x, y = float(pos[0]), float(pos[1])
        amp_kj, A_i, x0_i, y0_i, sx_i, sy_i, A_j, x0_j, y0_j, sx_j, sy_j = self._multiwell_params()

        U = float(amp_kj)  # flat offset

        # wells (negative Gaussians)
        dx = x - x0_i
        dy = y - y0_i
        arg = (dx * dx) / (2.0 * sx_i * sx_i) + (dy * dy) / (2.0 * sy_i * sy_i)
        U += float(-np.sum(A_i * np.exp(-arg)))

        # barrier (positive Gaussian)
        dxj = x - x0_j
        dyj = y - y0_j
        argj = (dxj * dxj) / (2.0 * sx_j * sx_j) + (dyj * dyj) / (2.0 * sy_j * sy_j)
        U += float(np.sum(A_j * np.exp(-argj)))

        # walls energy outside [0, 2π]
        k_wall = float(config.WALL_K)
        min_v = 0.0
        max_v = 2.0 * np.pi
        if x < min_v:
            U += k_wall * (min_v - x) ** 2
        elif x > max_v:
            U += k_wall * (x - max_v) ** 2
        if y < min_v:
            U += k_wall * (min_v - y) ** 2
        elif y > max_v:
            U += k_wall * (y - max_v) ** 2

        return float(U)

    def background_potential_grid(self, X, Y):
        amp_kj, A_i, x0_i, y0_i, sx_i, sy_i, A_j, x0_j, y0_j, sx_j, sy_j = self._multiwell_params()

        Z = np.full_like(X, float(amp_kj), dtype=float)

        dx = X[..., None] - x0_i
        dy = Y[..., None] - y0_i
        arg = (dx * dx) / (2.0 * sx_i * sx_i) + (dy * dy) / (2.0 * sy_i * sy_i)
        Z += -np.sum(A_i * np.exp(-arg), axis=-1)

        dxj = X[..., None] - x0_j
        dyj = Y[..., None] - y0_j
        argj = (dxj * dxj) / (2.0 * sx_j * sx_j) + (dyj * dyj) / (2.0 * sy_j * sy_j)
        Z += np.sum(A_j * np.exp(-argj), axis=-1)

        return Z

    def potential_force(self, pos):
        """FORCE = -∇U"""
        x, y = float(pos[0]), float(pos[1])
        _, A_i, x0_i, y0_i, sx_i, sy_i, A_j, x0_j, y0_j, sx_j, sy_j = self._multiwell_params()

        Fx = 0.0
        Fy = 0.0

        # wells: U = -A exp(-arg) => F = - A exp(-arg) * (dx/sx^2, dy/sy^2)
        dx = x - x0_i
        dy = y - y0_i
        arg = (dx * dx) / (2.0 * sx_i * sx_i) + (dy * dy) / (2.0 * sy_i * sy_i)
        e = np.exp(-arg)
        Fx += float(-np.sum(A_i * e * dx / (sx_i * sx_i)))
        Fy += float(-np.sum(A_i * e * dy / (sy_i * sy_i)))

        # barrier: U = +A exp(-arg) => F = + A exp(-arg) * (dx/sx^2, dy/sy^2)
        dxj = x - x0_j
        dyj = y - y0_j
        argj = (dxj * dxj) / (2.0 * sx_j * sx_j) + (dyj * dyj) / (2.0 * sy_j * sy_j)
        ej = np.exp(-argj)
        Fx += float(np.sum(A_j * ej * dxj / (sx_j * sx_j)))
        Fy += float(np.sum(A_j * ej * dyj / (sy_j * sy_j)))

        # walls force (consistent stiffness with potential)
        k_wall = float(config.WALL_K)
        min_v = 0.0
        max_v = 2.0 * np.pi
        if x < min_v:
            Fx += 2.0 * k_wall * (min_v - x)
        elif x > max_v:
            Fx -= 2.0 * k_wall * (x - max_v)
        if y < min_v:
            Fy += 2.0 * k_wall * (min_v - y)
        elif y > max_v:
            Fy -= 2.0 * k_wall * (y - max_v)

        return np.array([Fx, Fy], dtype=float)

    # -----------------------
    # Bias terms
    # -----------------------
    def bias_potential(self, pos, biases):
        if not getattr(config, "ENABLE_BIAS", True):
            return 0.0
        x, y = float(pos[0]), float(pos[1])
        E = 0.0
        for (A, x0, y0, sigma) in biases:
            if A == 0.0:
                continue
            sigma = max(float(sigma), 1e-6)
            dx = x - float(x0)
            dy = y - float(y0)
            E += float(A) * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        return float(E)

    def total_potential(self, pos, biases=None):
        if biases is None:
            biases = self.all_biases_in_episode
        return self.background_potential(pos) + self.bias_potential(pos, biases)

    def bias_potential_grid(self, X, Y, biases=None):
        if not getattr(config, "ENABLE_BIAS", True):
            return np.zeros_like(X, dtype=float)
        if biases is None:
            biases = self.all_biases_in_episode

        B = np.zeros_like(X, dtype=float)
        for (A, x0, y0, sigma) in biases:
            if A == 0.0:
                continue
            sigma = max(float(sigma), 1e-6)
            dx = X - float(x0)
            dy = Y - float(y0)
            B += float(A) * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        return B

    def bias_force(self, pos, biases, FMAX=200.0):
        if not getattr(config, "ENABLE_BIAS", True):
            return np.zeros(2, dtype=float)
        x, y = float(pos[0]), float(pos[1])
        total_force = np.zeros(2, dtype=float)

        for (A, x0, y0, sigma) in biases:
            if A == 0.0:
                continue
            sigma = max(float(sigma), 1e-6)
            dx = x - float(x0)
            dy = y - float(y0)
            ex = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            B = float(A) * ex
            coef = B / (sigma * sigma)
            total_force[0] += coef * dx
            total_force[1] += coef * dy

        fn = float(np.linalg.norm(total_force))
        if fn > FMAX:
            total_force *= (FMAX / (fn + 1e-12))
        return total_force

    # -----------------------
    # Step (XY target objective)
    # -----------------------
    def step(self, action_index):
        action_index = int(action_index)

        # choose action
        if action_index < len(self.action_tuples):
            amp, width, offset = self.action_tuples[action_index]
        else:
            amp, width, offset = 0.0, 1.0, 0.0

        # If already in target zone, force amp=0 (optional, keeps hill spam down)
        prev_dist = self.dist_to_target
        if prev_dist <= float(config.TARGET_RADIUS):
            amp = 0.0
        if not getattr(config, "ENABLE_BIAS", True):
            amp = 0.0

        # Place hill center based on bias placement mode
        if getattr(config, "ENABLE_BIAS", True):
            mode = getattr(config, "BIAS_PLACEMENT_MODE", "away_from_target")
            if mode == "current_position":
                # Metadynamics-style: deposit at current position
                x0, y0 = float(self.current_position[0]), float(self.current_position[1])
            elif mode == "away_from_target":
                # Heuristic: place hill away from target so repulsion pushes toward target
                target = np.array([float(config.TARGET_X), float(config.TARGET_Y)], dtype=float)
                vec_away = self.current_position - target
                norm = float(np.linalg.norm(vec_away))
                if norm < 1e-12:
                    theta = np.random.uniform(0, 2 * np.pi)
                    u_away = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                else:
                    u_away = vec_away / norm
                center_xy = self.current_position + float(offset) * u_away
                x0, y0 = float(center_xy[0]), float(center_xy[1])
            else:
                raise ValueError(f"Unknown BIAS_PLACEMENT_MODE: {mode}")

            if float(amp) != 0.0:
                self.all_biases_in_episode.append((float(amp), x0, y0, float(width)))
        if len(self.all_biases_in_episode) > config.MAX_BIASES:
            self.all_biases_in_episode = self.all_biases_in_episode[-config.MAX_BIASES:]

        # propagate overdamped Langevin
        n_steps = int(config.SIM_STEPS)
        dt = float(config.DT)
        temp = float(config.TEMPERATURE)
        gamma = float(config.FRICTION)

        D = temp / gamma
        sigma_noise = np.sqrt(2.0 * D * dt)

        traj_dist = []
        traj_xy = []

        for _ in range(n_steps):
            F = self.potential_force(self.current_position) + self.bias_force(self.current_position, self.all_biases_in_episode)
            if config.ENABLE_MILESTONE_LOCKS and self.milestones_reached:
                lock_dist = min(self.milestones_reached)
                F += self._lock_force(self.current_position, lock_dist)
            if config.ZONE_CONFINEMENT:
                F += self._zone_force(self.current_position)
            disp = (F / gamma) * dt + np.random.normal(0.0, 1.0, 2) * sigma_noise
            self.current_position += disp
            self._clip_to_domain()

            d = self._dist_to_target(self.current_position)
            traj_dist.append(d)
            traj_xy.append(self.current_position.copy())

        self.episode_trajectory_segments.append(traj_dist)
        self.episode_xy_segments.append(np.asarray(traj_xy, dtype=float))

        # compute new dist
        self.dist_to_target = float(traj_dist[-1])
        if self.dist_to_target <= float(config.TARGET_RADIUS):
            self.zone_confinement_active = True

        # reward: progress to target
        new_dist = self.dist_to_target
        progress = (prev_dist - new_dist)  # positive if closer
        reward = 0.0
        reward += float(config.PROGRESS_REWARD) * progress
        if progress < 0:
            reward += float(config.BACKTRACK_PENALTY)
        reward += float(config.STEP_PENALTY)
        if getattr(config, "ENABLE_BIAS", True):
            reward -= float(config.BIAS_PENALTY) * abs(float(amp))

        # milestone rewards based on distance-to-target thresholds
        for m in config.DISTANCE_INCREMENTS:
            if new_dist <= float(m) and m not in self.milestones_reached:
                reward += float(config.MILESTONE_REWARD)
                self.milestones_reached.add(m)

        # phase-2 stability bonus
        if new_dist <= float(config.PHASE2_TOL):
            self.stability_count += 1
            if self.stability_count == int(config.STABILITY_STEPS):
                reward += float(config.CONSISTENCY_BONUS)
        else:
            self.stability_count = 0

        done = False
        if new_dist <= float(config.TARGET_RADIUS):
            done = True
            reward += 200.0  # terminal bonus (tune if needed)

        return self.get_state(), reward, done, traj_dist

    # -----------------------
    # Helpers
    # -----------------------
    def _dist_to_target(self, pos):
        dx = float(pos[0]) - float(config.TARGET_X)
        dy = float(pos[1]) - float(config.TARGET_Y)
        return float(np.sqrt(dx * dx + dy * dy))

    def _clip_to_domain(self):
        min_v = 0.0
        max_v = 2.0 * np.pi
        # Reflect repeatedly so large overshoots still stay physical.
        for i in (0, 1):
            v = float(self.current_position[i])
            while v < min_v or v > max_v:
                if v < min_v:
                    v = min_v + (min_v - v)
                elif v > max_v:
                    v = max_v - (v - max_v)
            self.current_position[i] = float(np.clip(v, min_v, max_v))

    def _validate_action_bins(self):
        if any(a < config.MIN_AMP or a > config.MAX_AMP for a in config.AMP_BINS):
            raise ValueError("AMP_BINS out of bounds vs MIN_AMP/MAX_AMP")
        if any(w < config.MIN_WIDTH or w > config.MAX_WIDTH for w in config.WIDTH_BINS):
            raise ValueError("WIDTH_BINS out of bounds vs MIN_WIDTH/MAX_WIDTH")

    def _lock_force(self, pos, lock_dist):
        """Harmonic backstop if particle moves beyond a reached milestone."""
        dist = self._dist_to_target(pos)
        if dist <= lock_dist + float(config.LOCK_MARGIN):
            return np.zeros(2, dtype=float)
        target = np.array([float(config.TARGET_X), float(config.TARGET_Y)], dtype=float)
        vec_to_target = target - np.array([float(pos[0]), float(pos[1])], dtype=float)
        norm = float(np.linalg.norm(vec_to_target))
        if norm < 1e-12:
            return np.zeros(2, dtype=float)
        u = vec_to_target / norm
        # restoring force proportional to excess distance beyond lock_dist
        k = float(config.BACKSTOP_K)
        magnitude = k * (dist - (lock_dist + float(config.LOCK_MARGIN)))
        return u * magnitude

    def _zone_force(self, pos):
        """Soft confinement around the target only after the zone has been reached."""
        if not self.zone_confinement_active:
            return np.zeros(2, dtype=float)
        dist = self._dist_to_target(pos)
        max_allowed = float(config.TARGET_RADIUS) + float(config.ZONE_MARGIN_HIGH)
        if dist <= max_allowed:
            return np.zeros(2, dtype=float)
        target = np.array([float(config.TARGET_X), float(config.TARGET_Y)], dtype=float)
        vec_to_target = target - np.array([float(pos[0]), float(pos[1])], dtype=float)
        norm = float(np.linalg.norm(vec_to_target))
        if norm < 1e-12:
            return np.zeros(2, dtype=float)
        u = vec_to_target / norm
        k = float(config.ZONE_K)
        magnitude = k * (dist - max_allowed)
        return u * magnitude

    def _in_target_zone(self, pos):
        return self._dist_to_target(pos) <= float(config.TARGET_RADIUS)
