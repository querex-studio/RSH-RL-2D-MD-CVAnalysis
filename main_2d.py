import os
import numpy as np
import torch
import combined_2d as config
from combined_2d import (
    PPOAgent,
    ProteinEnvironmentRedesigned,
    _ensure,
    save_checkpoint,
    plot_distance_trajectory,
    save_episode_bias_profiles,
    write_episode_pdb,
)
from torch.serialization import add_safe_globals
import openmm.unit as mm_unit
import torch
from torch.serialization import add_safe_globals
import openmm.unit as mm_unit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend that writes files only

WARMUP_EPISODES = 10  # small warm-up with locks to validate stability

# Allow-list OpenMM classes used inside the checkpoint
add_safe_globals([mm_unit.quantity.Quantity, mm_unit.unit.Unit])

def load_agent_from_episode(ep_idx: int):
    results_dir = _ensure(config.RESULTS_DIR)
    ckpt_dir = _ensure(os.path.join(results_dir, "checkpoints"))
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep_{ep_idx:04d}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found for episode {ep_idx} at {ckpt_path}")

    # First try the safe weights-only loader
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fallback: legacy behavior (only do this if you trust the file, which you do)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    agent = PPOAgent(config.STATE_SIZE, config.ACTION_SIZE, config.SEED)
    agent.actor.load_state_dict(payload["actor"])
    agent.critic.load_state_dict(payload["critic"])
    if "obs_norm" in payload:
        agent.load_obs_norm_state(payload["obs_norm"])

    print(f"[resume] Loaded checkpoint from episode {ep_idx}: {ckpt_path}")
    return agent

def train_progressive(n_episodes=400, start_ep=1, agent=None):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # re-use agent if passed (for resume), otherwise start from scratch
    if agent is None:
        agent = PPOAgent(config.STATE_SIZE, config.ACTION_SIZE, config.SEED)

    env = ProteinEnvironmentRedesigned()

    results_dir = _ensure(config.RESULTS_DIR)
    ckpt_dir = _ensure(os.path.join(results_dir, "checkpoints"))

    # start_ep lets us resume from a later episode number
    for ep_idx in range(start_ep, start_ep + n_episodes):
        # warm-up schedule then true training
        if ep_idx <= WARMUP_EPISODES:
            config.ENABLE_MILESTONE_LOCKS = True
            config.IN_ZONE_MAX_AMP = 1.0
        else:
            config.ENABLE_MILESTONE_LOCKS = False
            config.IN_ZONE_MAX_AMP = 1e9

        # fresh or curriculum start
        state = env.reset(seed_from_max_A=None,
                          carry_state=False,
                          episode_index=ep_idx)
        done = False
        steps = 0

        while not done and steps < config.MAX_ACTIONS_PER_EPISODE:
            action, logp, value = agent.act(state, training=True)
            next_state, reward, done, dists = env.step(action)
            agent.save_experience(
                state, action, logp, value, reward, done, next_state
            )
            state = next_state
            steps += 1

        metrics = agent.update()

        # checkpoint
        if ep_idx % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(agent, env, ckpt_dir, ep_idx)

        # plotting fallback if no segments were recorded
        if not getattr(env, "episode_trajectory_segments", []):
            if len(getattr(env, "distance_history", [])) > 1:
                env.episode_trajectory_segments = [
                    list(map(float, env.distance_history[1:]))
                ]

        if not getattr(env, "episode_trajectory_segments_cv2", []):
            if len(getattr(env, "distance2_history", [])) > 1:
                env.episode_trajectory_segments_cv2 = [
                    list(map(float, env.distance2_history[1:]))
                ]

        # plot trajectory
        if getattr(env, "episode_trajectory_segments", []):
            plot_distance_trajectory(
                env.episode_trajectory_segments,
                ep_idx,
                distance_history=getattr(env, "distance_history", None),
                episode_trajectories_cv2=getattr(env, "episode_trajectory_segments_cv2", None),
            )
        else:
            print(
                f"[Warning] No trajectory data recorded for episode "
                f"{ep_idx}. Nothing to plot."
            )

        # bias profile per episode
        if getattr(config, "SAVE_BIAS_PROFILE", False):
            if (ep_idx % int(getattr(config, "BIAS_PROFILE_EVERY", 1))) == 0:
                save_episode_bias_profiles(getattr(env, "all_biases_in_episode", []), ep_idx)

        # save PDB per episode
        if getattr(config, "SAVE_EPISODE_PDB", False):
            if (ep_idx % int(getattr(config, "EPISODE_PDB_EVERY", 1))) == 0:
                write_episode_pdb(env, getattr(config, "EPISODE_PDB_DIR", config.RESULTS_DIR), ep_idx)

        # simple console log
        if metrics:
            print(
                f"[ep {ep_idx:04d}] steps={steps} "
                f"loss={metrics.get('loss', 0):.3f} "
                f"actor={metrics.get('actor_loss', 0):.3f} "
                f"critic={metrics.get('critic_loss', 0):.3f} "
                f"ent={metrics.get('entropy', 0):.3f} "
                f"kl={metrics.get('approx_kl', 0):.3f} "
                f"clip={metrics.get('clip_frac', 0):.2f} "
                f"lr={metrics.get('lr', 0):.2e}"
            )

    print("Training complete.")


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Run control: start fresh OR resume from an on-disk checkpoint.
    #
    # Set RESUME=False to start fresh training from episode 1 with a new agent.
    # Set RESUME=True  to resume from `resume_ep` if that checkpoint exists.
    # If the checkpoint is missing, it will automatically fall back to fresh.
    # ---------------------------------------------------------------------

    RESUME = False          # True -> resume from checkpoint; False -> start fresh
    resume_ep = 295        # checkpoint episode index to load (only used if RESUME=True)
    total_target = 5     # total episodes you want to complete in this run

    if RESUME:
        try:
            agent = load_agent_from_episode(resume_ep)
            remaining = total_target - resume_ep
            start_ep = resume_ep + 1
        except FileNotFoundError as e:
            print(f"[resume] {e}")
            print("[resume] Falling back to fresh training from episode 1.")
            agent = None
            remaining = total_target
            start_ep = 1
    else:
        agent = None
        remaining = total_target
        start_ep = 1

    train_progressive(
        n_episodes=remaining,
        start_ep=start_ep,
        agent=agent,
    )
