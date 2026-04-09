from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from comparison_config import EnvironmentConfig, PPOConfig
from comparison_encoders import BaseEncoder


class RunningNorm:
    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.var = None
        self.count = eps

    def update(self, x: np.ndarray | torch.Tensor) -> None:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        if self.mean is None:
            self.mean = batch_mean
            self.var = np.maximum(batch_var, 1e-8)
            self.count = batch_count
            return
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = np.maximum(m2 / total_count, 1e-8)
        self.count = total_count

    def normalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.mean is None or self.var is None:
            return x
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(np.sqrt(self.var) + 1e-8, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class MLPActor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        h1, h2, h3 = hidden_sizes
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, output_dim)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        return torch.clamp(self.fc4(x), -20.0, 20.0)


class MLPCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        h1, h2, h3 = hidden_sizes
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        return self.fc4(x)


@dataclass
class PPOTransition:
    raw_state: np.ndarray
    encoded_state: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    next_raw_state: np.ndarray
    next_encoded_state: np.ndarray


class PPOAgent:
    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        ppo_cfg: PPOConfig,
        encoder: BaseEncoder,
        seed: int,
    ):
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg
        self.encoder = encoder
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.action_map = [
            (a, w, o)
            for a in env_cfg.amp_bins
            for w in env_cfg.width_bins
            for o in env_cfg.offset_bins
        ]
        input_dim = encoder.output_dim
        self.actor = MLPActor(input_dim, env_cfg.action_size, ppo_cfg.hidden_sizes)
        self.critic = MLPCritic(input_dim, ppo_cfg.hidden_sizes)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=ppo_cfg.lr,
            eps=1e-5,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=ppo_cfg.scheduler_step,
            gamma=ppo_cfg.scheduler_gamma,
        )
        self.obs_rms = RunningNorm()
        self.memory: List[PPOTransition] = []
        self.exploration_noise = ppo_cfg.exploration_noise

    def encode(self, raw_state: np.ndarray) -> np.ndarray:
        enc = self.encoder.transform(raw_state).astype(np.float32)
        return enc

    def act(self, raw_state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        enc_state = self.encode(raw_state)
        enc_batch = enc_state[None, :]
        if training:
            self.obs_rms.update(enc_batch)
        elif self.obs_rms.mean is None:
            self.obs_rms.update(enc_batch)
        norm_state = torch.from_numpy(self.obs_rms.normalize(enc_batch).astype(np.float32))
        with torch.no_grad():
            logits = self.actor.forward_logits(norm_state)
            logits = self._mask_logits(logits, raw_state[None, :])
            if training and self.exploration_noise > self.ppo_cfg.min_exploration_noise:
                if not self._freeze_exploration(raw_state):
                    logits = logits + torch.randn_like(logits) * self.exploration_noise
            dist = torch.distributions.Categorical(logits=logits)
            probs = dist.probs
            value = float(self.critic(norm_state).item())
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]
            dist = torch.distributions.Categorical(probs=probs)
        if (not training) and self.ppo_cfg.eval_greedy:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), value

    def remember(
        self,
        raw_state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        next_raw_state: np.ndarray,
    ) -> None:
        encoded_state = self.encode(raw_state)
        next_encoded_state = self.encode(next_raw_state)
        self.memory.append(
            PPOTransition(
                raw_state=np.asarray(raw_state, dtype=np.float32),
                encoded_state=np.asarray(encoded_state, dtype=np.float32),
                action=int(action),
                log_prob=float(log_prob),
                value=float(value),
                reward=float(reward),
                done=bool(done),
                next_raw_state=np.asarray(next_raw_state, dtype=np.float32),
                next_encoded_state=np.asarray(next_encoded_state, dtype=np.float32),
            )
        )

    def update(self) -> Dict[str, float]:
        if len(self.memory) < self.ppo_cfg.n_steps:
            return {}
        data = self._compute_advantages()
        if data is None:
            return {}
        raw_states, states, actions, old_log_probs, advantages, returns = data
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n_items = len(self.memory)
        metrics = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "updates": 0,
        }
        stop_early = False
        for _ in range(self.ppo_cfg.n_epochs):
            perm = torch.randperm(n_items)
            for start in range(0, n_items, self.ppo_cfg.batch_size):
                idx = perm[start : start + self.ppo_cfg.batch_size]
                if len(idx) < 2:
                    continue
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_raw_states = raw_states[idx.cpu().numpy()]
                batch_old = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                logits = self.actor.forward_logits(batch_states)
                logits = self._mask_logits(logits, batch_raw_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratios = torch.exp(new_logp - batch_old)
                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1 - self.ppo_cfg.clip_range, 1 + self.ppo_cfg.clip_range) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = f.mse_loss(values, batch_ret)
                loss = actor_loss + self.ppo_cfg.vf_coef * critic_loss - self.ppo_cfg.ent_coef * entropy
                approx_kl = float((batch_old - new_logp).mean().item())
                clip_frac = float((torch.abs(ratios - 1.0) > self.ppo_cfg.clip_range).float().mean().item())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo_cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo_cfg.max_grad_norm)
                self.optimizer.step()
                metrics["loss"] += float(loss.item())
                metrics["actor_loss"] += float(actor_loss.item())
                metrics["critic_loss"] += float(critic_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac
                metrics["updates"] += 1
                if approx_kl > self.ppo_cfg.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        self.scheduler.step()
        self.exploration_noise = max(
            self.ppo_cfg.min_exploration_noise,
            self.exploration_noise * self.ppo_cfg.exploration_decay,
        )
        self.memory = []
        if metrics["updates"] > 0:
            for key in list(metrics.keys()):
                if key != "updates":
                    metrics[key] /= metrics["updates"]
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    def _compute_advantages(self):
        if not self.memory:
            return None
        raw_states = np.array([item.raw_state for item in self.memory], dtype=np.float32)
        encoded_states = np.array([item.encoded_state for item in self.memory], dtype=np.float32)
        actions = torch.tensor([item.action for item in self.memory], dtype=torch.long)
        old_log_probs = torch.tensor([item.log_prob for item in self.memory], dtype=torch.float32)
        values = torch.tensor([item.value for item in self.memory], dtype=torch.float32)
        rewards = torch.tensor([item.reward for item in self.memory], dtype=torch.float32)
        dones = torch.tensor([float(item.done) for item in self.memory], dtype=torch.float32)
        norm_states = torch.from_numpy(self.obs_rms.normalize(encoded_states).astype(np.float32))
        if self.memory[-1].done:
            next_value = 0.0
        else:
            next_state = np.asarray(self.memory[-1].next_encoded_state, dtype=np.float32)[None, :]
            next_state = torch.from_numpy(self.obs_rms.normalize(next_state).astype(np.float32))
            with torch.no_grad():
                next_value = float(self.critic(next_state).item())
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        next_values = torch.zeros_like(values)
        if len(values) > 1:
            next_values[:-1] = values[1:]
        next_values[-1] = next_value
        gae = 0.0
        for t in reversed(range(len(rewards))):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.ppo_cfg.gamma * next_values[t] * not_done - values[t]
            gae = delta + self.ppo_cfg.gamma * self.ppo_cfg.gae_lambda * not_done * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return raw_states, norm_states, actions, old_log_probs, advantages, returns

    def _freeze_exploration(self, raw_state: np.ndarray) -> bool:
        in_zone = float(raw_state[3]) >= 0.5
        return bool(self.env_cfg.free_exploration_at_zone and in_zone)

    def _mask_logits(self, logits: torch.Tensor, raw_states: np.ndarray) -> torch.Tensor:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
            raw_states = raw_states[None, :]
        mask_values = []
        for state in np.asarray(raw_states, dtype=np.float32):
            in_zone = float(state[3]) >= 0.5
            if not in_zone:
                mask_values.append(np.zeros(self.env_cfg.action_size, dtype=np.float32))
                continue
            mask = np.zeros(self.env_cfg.action_size, dtype=np.float32)
            for idx, (amp, _, _) in enumerate(self.action_map):
                if amp > self.env_cfg.in_zone_max_amp:
                    mask[idx] = -1e9
            mask_values.append(mask)
        mask_tensor = torch.tensor(np.stack(mask_values), dtype=logits.dtype, device=logits.device)
        return logits + mask_tensor
