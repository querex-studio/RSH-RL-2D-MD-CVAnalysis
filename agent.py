# agent.py — PPO Agent (final)
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config_gaussian as config

class RunningNorm:
    def __init__(self, eps=1e-8):
        self.mean = None; self.var = None; self.count = eps
    def update(self, x):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1: x = x[None, :]
        b_mean = x.mean(axis=0); b_var = x.var(axis=0); b_cnt = x.shape[0]
        if self.mean is None:
            self.mean, self.var, self.count = b_mean, np.maximum(b_var, 1e-8), b_cnt
        else:
            delta = b_mean - self.mean; tot = self.count + b_cnt
            new_mean = self.mean + delta * b_cnt / tot
            m_a = self.var * self.count; m_b = b_var * b_cnt
            M2 = m_a + m_b + delta**2 * self.count * b_cnt / tot
            new_var = np.maximum(M2 / tot, 1e-8)
            self.mean, self.var, self.count = new_mean, new_var, tot
    def normalize(self, x):
        if self.mean is None or self.var is None: return x
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std  = torch.tensor(np.sqrt(self.var)+1e-8, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
    def state_dict(self):
        return {'mean': None if self.mean is None else self.mean.tolist(),
                'var': None if self.var is None else self.var.tolist(),
                'count': float(self.count)}
    def load_state_dict(self, d):
        if not d: return
        self.mean  = None if d.get('mean') is None else np.asarray(d['mean'], dtype=np.float64)
        self.var   = None if d.get('var')  is None else np.asarray(d['var'], dtype=np.float64)
        self.count = float(d.get('count', 1e-8))

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1=256, fc2=128, fc3=64):
        super().__init__(); torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1); self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3); self.fc4 = nn.Linear(fc3, action_size)
    def forward_logits(self, state):
        x = F.relu(self.fc1(state)); x = F.relu(self.fc2(x)); x = F.relu(self.fc3(x))
        return torch.clamp(self.fc4(x), -20, 20)
    def forward(self, state):  # not used directly
        return F.softmax(self.forward_logits(state), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size, seed, fc1=256, fc2=128, fc3=64):
        super().__init__(); torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1); self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3); self.fc4 = nn.Linear(fc3, 1)
    def forward(self, state):
        x = F.relu(self.fc1(state)); x = F.relu(self.fc2(x)); x = F.relu(self.fc3(x))
        return self.fc4(x)

class PPOAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size; self.action_size = action_size; self.seed = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.action_map = [(A, W, O) for A in config.AMP_BINS
                                     for W in config.WIDTH_BINS
                                     for O in config.OFFSET_BINS]

        self.actor = Actor(state_size, action_size, seed)
        self.critic = Critic(state_size, seed)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.LR, eps=1e-5
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

        self.memory = []
        self.gamma = config.GAMMA; self.gae_lambda = config.GAE_LAMBDA
        self.clip_range = config.CLIP_RANGE; self.n_epochs = config.N_EPOCHS
        self.batch_size = config.BATCH_SIZE; self.ent_coef = config.ENT_COEF
        self.vf_coef = config.VF_COEF; self.max_grad_norm = config.MAX_GRAD_NORM
        self.target_kl = config.PPO_TARGET_KL

        self.exploration_noise = 0.05; self.min_exploration_noise = 0.005
        self.exploration_decay = 0.99

        self.obs_rms = RunningNorm()
        self.episode_count = 0

    def _mask_logits_if_needed(self, logits, raw_state_batch):
        if logits.ndim == 1:
            logits = logits.unsqueeze(0); raw_state_batch = raw_state_batch[None, :]
        mask_vals = []
        for s in np.asarray(raw_state_batch, dtype=np.float32):
            in_zone = (s[config.STATE_IDX_IN_ZONE] >= 0.5)
            if not in_zone:
                mask_vals.append(np.zeros(self.action_size, dtype=np.float32))
            else:
                m = np.zeros(self.action_size, dtype=np.float32)
                for idx, (A, _, _) in enumerate(self.action_map):
                    if A > config.IN_ZONE_MAX_AMP:
                        m[idx] = -1e9
                mask_vals.append(m)
        mask = torch.tensor(np.stack(mask_vals), dtype=logits.dtype, device=logits.device)
        return logits + mask

    def act(self, state, training=True):
        s_np = np.asarray(state, dtype=np.float32)
        if s_np.ndim == 1: s_np = s_np[None, :]
        if training:
            self.obs_rms.update(s_np)
        elif self.obs_rms.mean is None:
            self.obs_rms.update(s_np)
        norm_np = self.obs_rms.normalize(s_np).astype(np.float32)
        norm_state = torch.from_numpy(norm_np)

        with torch.no_grad():
            logits = self.actor.forward_logits(norm_state)
            logits = self._mask_logits_if_needed(logits, s_np)
            if training:
                in_zone = (float(s_np[0, config.STATE_IDX_IN_ZONE]) >= 0.5)
                freeze = (config.FREEZE_EXPLORATION_AT_ZONE and in_zone)
                if self.exploration_noise > self.min_exploration_noise and not freeze:
                    logits = logits + torch.randn_like(logits) * self.exploration_noise
            dist = torch.distributions.Categorical(logits=logits)
            probs = dist.probs
            value = self.critic(norm_state).item()

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / self.action_size
            dist = torch.distributions.Categorical(probs=probs)

        if not training and config.EVAL_GREEDY:
            action = torch.argmax(probs, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample(); log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value)

    def save_experience(self, s, a, logp, v, r, done, ns):
        self.memory.append((s, a, logp, v, r, done, ns))

    def compute_advantages(self):
        if not self.memory: return None
        states = np.array([m[0] for m in self.memory], dtype=np.float32)
        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.long)
        old_log_probs = torch.tensor([m[2] for m in self.memory], dtype=torch.float32)
        values = torch.tensor([m[3] for m in self.memory], dtype=torch.float32)
        rewards = torch.tensor([m[4] for m in self.memory], dtype=torch.float32)
        dones = torch.tensor([m[5] for m in self.memory], dtype=torch.float32)

        norm_states = torch.from_numpy(self.obs_rms.normalize(states)).float()

        if self.memory[-1][5]:
            next_value = 0.0
        else:
            next_state = np.asarray(self.memory[-1][6], dtype=np.float32)
            if next_state.ndim == 1: next_state = next_state[None, :]
            ns = self.obs_rms.normalize(next_state).astype(np.float32)
            with torch.no_grad():
                next_value = self.critic(torch.from_numpy(ns)).item()

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        next_values = torch.zeros_like(values)
        if len(values) > 1:
            next_values[:-1] = values[1:]
        next_values[-1] = next_value
        gae = 0.0
        for t in reversed(range(len(rewards))):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae; returns[t] = gae + values[t]

        return states, norm_states, actions, old_log_probs, advantages, returns

    def update(self):
        if len(self.memory) < config.N_STEPS: return {}
        data = self.compute_advantages()
        if data is None: return {}
        raw_states, states, actions, old_log_probs, advantages, returns = data
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(self.memory)
        metrics = {'loss':0.0,'actor_loss':0.0,'critic_loss':0.0,'entropy':0.0,
                   'approx_kl':0.0,'clip_frac':0.0,'updates':0}

        stop_early = False
        for _ in range(config.N_EPOCHS):
            perm = torch.randperm(N)
            for i in range(0, N, config.BATCH_SIZE):
                idx = perm[i:i+config.BATCH_SIZE]
                if len(idx) < 2: continue
                b_states = states[idx]; b_actions = actions[idx]
                b_raw_states = raw_states[idx.cpu().numpy()]
                b_old = old_log_probs[idx]; b_adv = advantages[idx]; b_ret = returns[idx]

                logits = self.actor.forward_logits(b_states)
                logits = self._mask_logits_if_needed(logits, b_raw_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_logp - b_old)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(b_states).squeeze()
                critic_loss = F.mse_loss(values, b_ret)

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                approx_kl = (b_old - new_logp).mean().item()
                clip_frac = (torch.abs(ratios - 1.0) > self.clip_range).float().mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['loss'] += loss.item()
                metrics['actor_loss'] += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += approx_kl
                metrics['clip_frac'] += clip_frac
                metrics['updates'] += 1

                if approx_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.scheduler.step()
        self.exploration_noise = max(self.min_exploration_noise,
                                     self.exploration_noise * self.exploration_decay)
        self.memory = []
        self.episode_count += 1

        if metrics['updates'] > 0:
            for k in list(metrics.keys()):
                if k not in ('updates',): metrics[k] /= metrics['updates']
        metrics['lr'] = float(self.optimizer.param_groups[0]['lr'])
        return metrics

    def obs_norm_state(self): return self.obs_rms.state_dict()
    def load_obs_norm_state(self, d): self.obs_rms.load_state_dict(d)
