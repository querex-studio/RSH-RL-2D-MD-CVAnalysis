import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config_gaussian
from agent import PPOAgent
from env_gaussian_2d import Gaussian2DEnvironment

_LANDSCAPE_CACHE = {}


def make_landscape_grid(env, grid_n=None):
    if grid_n is None:
        grid_n = config_gaussian.PLOT_GRID_SIZE

    if grid_n not in _LANDSCAPE_CACHE:
        min_v = 0.0
        max_v = 2.0 * np.pi
        xs = np.linspace(min_v, max_v, grid_n)
        ys = np.linspace(min_v, max_v, grid_n)
        X, Y = np.meshgrid(xs, ys)
        Z = env.background_potential_grid(X, Y)
        _LANDSCAPE_CACHE[grid_n] = (X, Y, Z, min_v, max_v)

    X, Y, Z, min_v, max_v = _LANDSCAPE_CACHE[grid_n]
    B = env.bias_potential_grid(X, Y)
    return X, Y, Z, B, min_v, max_v


def plot_initial_potential(env, out_path=None):
    if out_path is None:
        out_path = config_gaussian.INITIAL_POTENTIAL_PLOT

    X, Y, Z, _, min_v, max_v = make_landscape_grid(env)
    fig, ax = plt.subplots(figsize=(7, 6))

    lo, hi = np.percentile(Z, [2, 98])
    if hi <= lo:
        lo = float(Z.min())
        hi = float(Z.max()) + 1e-6

    cf = ax.contourf(X, Y, Z, levels=80, cmap="winter", norm=Normalize(vmin=lo, vmax=hi))
    ax.contour(X, Y, Z, levels=22, colors="k", alpha=0.55, linewidths=0.9)

    ax.plot(config_gaussian.START_X, config_gaussian.START_Y, "go", markersize=8, label="Start")
    ax.plot(config_gaussian.TARGET_X, config_gaussian.TARGET_Y, "r*", markersize=12, label="Target")
    ax.add_patch(
        plt.Circle(
            (config_gaussian.TARGET_X, config_gaussian.TARGET_Y),
            config_gaussian.TARGET_RADIUS,
            color="r",
            fill=False,
            linestyle="--",
            linewidth=1.2,
        )
    )

    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Potential")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Initial Potential Landscape")
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2d_trajectory(env, episode_num, out_dir=None):
    if out_dir is None:
        out_dir = config_gaussian.PLOTS_DIR
    if not env.episode_xy_segments:
        return

    xy_data = np.concatenate(env.episode_xy_segments, axis=0)
    if xy_data.ndim != 2 or xy_data.shape[1] != 2:
        return

    X, Y, Z, B, min_v, max_v = make_landscape_grid(env)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal", adjustable="box")
    success = bool(env.dist_to_target <= float(config_gaussian.TARGET_RADIUS))

    lo, hi = np.percentile(Z, [2, 98])
    if hi <= lo:
        lo = float(Z.min())
        hi = float(Z.max()) + 1e-6
    norm_psp = Normalize(vmin=lo, vmax=hi)

    cf = ax.contourf(X, Y, Z, levels=80, cmap="winter", alpha=0.65, norm=norm_psp, zorder=0)
    ax.contour(X, Y, Z, levels=22, colors="k", alpha=0.65, linewidths=1.0, zorder=1)

    bias_im = None
    if np.any(B > 0):
        vmax_b = np.percentile(B[B > 0], 98)
        if vmax_b <= 0:
            vmax_b = float(B.max()) + 1e-12
        bias_im = ax.imshow(
            B,
            extent=[min_v, max_v, min_v, max_v],
            origin="lower",
            cmap="OrRd",
            alpha=0.60,
            norm=PowerNorm(gamma=0.9, vmin=0.0, vmax=vmax_b, clip=True),
            interpolation="none",
            zorder=2,
        )

    divider = make_axes_locatable(ax)
    cax_bias = divider.append_axes("right", size="4.5%", pad=0.5)
    cax_psp = divider.append_axes("right", size="4.5%", pad=0.5)
    fig.subplots_adjust(right=0.9)

    if bias_im is not None:
        cb_bias = fig.colorbar(bias_im, cax=cax_bias)
        cb_bias.set_label("Bias (sum of hills)", fontsize=9, labelpad=4)
        cb_bias.ax.yaxis.set_ticks_position("left")
        cb_bias.ax.yaxis.set_label_position("right")
        cb_bias.locator = MaxNLocator(nbins=7)
        cb_bias.update_ticks()
        cb_bias.set_alpha(0.3)
        fig.canvas.draw_idle()
        cb_bias.ax.tick_params(labelsize=9)
    else:
        cax_bias.remove()

    cb_psp = fig.colorbar(cf, cax=cax_psp)
    cb_psp.set_label("Potential", fontsize=9, labelpad=4)
    cb_psp.ax.yaxis.set_ticks_position("left")
    cb_psp.ax.yaxis.set_label_position("right")
    cb_psp.locator = MaxNLocator(nbins=7)
    cb_psp.update_ticks()
    cb_psp.set_alpha(0.3)
    fig.canvas.draw_idle()
    cb_psp.ax.tick_params(labelsize=9)

    ax.plot(xy_data[:, 0], xy_data[:, 1], "k-", alpha=0.65, linewidth=1.0, zorder=7)
    ax.plot(config_gaussian.START_X, config_gaussian.START_Y, "go", label="Start", alpha=0.8, zorder=8)
    ax.plot(config_gaussian.TARGET_X, config_gaussian.TARGET_Y, "r*", markersize=10, label="Target", zorder=8)
    ax.plot(
        xy_data[-1, 0],
        xy_data[-1, 1],
        marker="o" if success else "X",
        color="limegreen" if success else "darkorange",
        markersize=9,
        label="Final point",
        zorder=9,
    )
    ax.add_patch(
        plt.Circle(
            (config_gaussian.TARGET_X, config_gaussian.TARGET_Y),
            config_gaussian.TARGET_RADIUS,
            color="r",
            fill=False,
            linestyle="--",
            linewidth=1.2,
            zorder=8,
        )
    )

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    status_label = "HIT target" if success else "MISS target"
    ax.set_title(
        f"Episode {episode_num} | {status_label} | final distance={env.dist_to_target:.3f} | "
        f"steps={xy_data.shape[0]}"
    )
    ax.text(
        0.02,
        0.98,
        status_label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="darkgreen" if success else "darkred",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        zorder=10,
    )
    ax.legend(loc="upper right")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"traj_ep_{episode_num:04d}.png")
    fig.savefig(out_path, dpi=config_gaussian.TRAJECTORY_DPI, bbox_inches="tight")
    plt.close(fig)


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=float) / window
    averaged = np.convolve(values, kernel, mode="valid")
    prefix = np.full(len(values) - len(averaged), np.nan, dtype=float)
    return np.concatenate([prefix, averaged])


def write_metrics_csv(history, out_path):
    if not history:
        return

    fieldnames = []
    seen = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def load_history_from_csv(csv_path):
    history = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value in ("", None):
                    parsed[key] = np.nan
                    continue
                try:
                    numeric = float(value)
                    if numeric.is_integer():
                        parsed[key] = int(numeric)
                    else:
                        parsed[key] = numeric
                except ValueError:
                    parsed[key] = value
            history.append(parsed)
    return history


def plot_training_dashboard(history, out_path=None):
    if out_path is None:
        out_path = config_gaussian.METRICS_PLOT
    if not history:
        return

    episodes = np.array([row["episode"] for row in history], dtype=float)
    scores = np.array([row["score"] for row in history], dtype=float)
    successes = np.array([row["success"] for row in history], dtype=float)
    final_dist = np.array([row["final_distance"] for row in history], dtype=float)
    steps = np.array([row["steps"] for row in history], dtype=float)
    n_biases = np.array([row["n_biases"] for row in history], dtype=float)
    actor_loss = np.array([row["actor_loss"] for row in history], dtype=float)
    critic_loss = np.array([row["critic_loss"] for row in history], dtype=float)
    entropy = np.array([row["entropy"] for row in history], dtype=float)
    approx_kl = np.array([row["approx_kl"] for row in history], dtype=float)
    clip_frac = np.array([row["clip_frac"] for row in history], dtype=float)
    lr = np.array([row["lr"] for row in history], dtype=float)
    eval_success = np.array([row.get("eval_success_rate", np.nan) for row in history], dtype=float)
    eval_score = np.array([row.get("eval_score_mean", np.nan) for row in history], dtype=float)
    eval_final_dist = np.array([row.get("eval_final_distance_mean", np.nan) for row in history], dtype=float)

    score_ma = moving_average(scores, config_gaussian.MOVING_AVG_WINDOW)
    success_ma = moving_average(successes, config_gaussian.MOVING_AVG_WINDOW)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()

    axes[0].plot(episodes, scores, color="steelblue", alpha=0.35, label="Episode score")
    axes[0].plot(episodes, score_ma, color="navy", linewidth=2.0, label="Moving avg")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].legend(loc="best")

    axes[1].plot(episodes, successes, color="seagreen", alpha=0.15, label="Episode success")
    axes[1].plot(episodes, success_ma, color="darkgreen", linewidth=2.0, label="Train success rate")
    if np.any(eval_mask := np.isfinite(eval_success)):
        axes[1].plot(episodes[eval_mask], eval_success[eval_mask], marker="o", color="purple", label="Eval success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Success Rate")
    axes[1].set_xlabel("Episode")
    axes[1].legend(loc="best")

    if np.any(eval_mask):
        axes[2].plot(episodes[eval_mask], eval_score[eval_mask], marker="s", color="darkorange", label="Eval score")
        axes[2].plot(episodes[eval_mask], eval_final_dist[eval_mask], marker="o", color="firebrick", label="Eval final distance")
    else:
        axes[2].text(0.5, 0.5, "No eval points yet", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].plot(episodes, final_dist, color="steelblue", alpha=0.85, label="Train final distance")
    axes[2].set_title("Final Distance and Eval Score")
    axes[2].set_xlabel("Episode")
    axes[2].legend(loc="best")

    axes[3].plot(episodes, actor_loss, color="tab:blue", label="Actor loss")
    axes[3].plot(episodes, critic_loss, color="tab:red", label="Critic loss")
    axes[3].plot(episodes, entropy, color="tab:green", label="Entropy")
    axes[3].set_title("Optimization")
    axes[3].set_xlabel("Episode")
    axes[3].legend(loc="best")

    axes[4].plot(episodes, approx_kl, color="tab:purple", label="Approx KL")
    axes[4].plot(episodes, clip_frac, color="tab:brown", label="Clip fraction")
    axes[4].plot(episodes, lr, color="black", alpha=0.6, label="LR")
    axes[4].set_title("PPO Diagnostics")
    axes[4].set_xlabel("Episode")
    axes[4].legend(loc="best")

    axes[5].plot(episodes, steps, color="tab:cyan", label="Steps")
    axes[5].plot(episodes, n_biases, color="tab:orange", label="Biases used")
    axes[5].set_title("Episode Length and Bias Count")
    axes[5].set_xlabel("Episode")
    axes[5].legend(loc="best")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_analysis(history, out_path=None):
    if out_path is None:
        out_path = config_gaussian.SUCCESS_PLOT
    if not history:
        return

    episodes = np.array([row["episode"] for row in history], dtype=float)
    successes = np.array([row["success"] for row in history], dtype=float)
    train_rate = moving_average(successes, config_gaussian.MOVING_AVG_WINDOW)
    cumulative_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
    eval_success = np.array([row.get("eval_success_rate", np.nan) for row in history], dtype=float)
    eval_mask = np.isfinite(eval_success)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(episodes, successes, color="lightgray", alpha=0.25, label="Episode success")
    ax.plot(episodes, train_rate, color="darkgreen", linewidth=2.2, label=f"Rolling success rate ({config_gaussian.MOVING_AVG_WINDOW})")
    ax.plot(episodes, cumulative_rate, color="navy", linewidth=1.8, label="Cumulative success rate")
    if np.any(eval_mask):
        ax.plot(episodes[eval_mask], eval_success[eval_mask], marker="o", color="purple", linewidth=1.8, label="Eval success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    ax.set_title("Success Rate Analysis")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(agent, path, extra=None):
    payload = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": agent.scheduler.state_dict(),
        "obs_rms": agent.obs_norm_state(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def evaluate(agent, n_eps=5, train_ep=0, seed_base=None):
    if seed_base is None:
        seed_base = config_gaussian.SEED

    env = Gaussian2DEnvironment()
    os.makedirs(config_gaussian.EVAL_PLOTS_DIR, exist_ok=True)

    scores = []
    steps_list = []
    success = 0
    final_distances = []

    for k in range(n_eps):
        np.random.seed(seed_base + k)
        state = env.reset(carry_state=False, episode_index=10_000 + k)
        done = False
        steps = 0
        score = 0.0

        while not done and steps < config_gaussian.MAX_ACTIONS_PER_EPISODE:
            action, _, _ = agent.act(state, training=False)
            state, reward, done, _ = env.step(action)
            score += reward
            steps += 1

        scores.append(score)
        steps_list.append(steps)
        final_distances.append(env.dist_to_target)
        if done:
            success += 1

        np.savetxt(
            os.path.join(config_gaussian.EVAL_PLOTS_DIR, f"traj_ep_{train_ep:04d}_roll_{k:02d}.csv"),
            np.concatenate(env.episode_xy_segments, axis=0),
            delimiter=",",
            header="x,y",
            comments="",
        )
        plot_2d_trajectory(env, episode_num=train_ep + k, out_dir=config_gaussian.EVAL_PLOTS_DIR)

    return {
        "eval_score_mean": float(np.mean(scores)),
        "eval_score_std": float(np.std(scores)),
        "eval_steps_mean": float(np.mean(steps_list)),
        "eval_success_rate": float(success / n_eps),
        "eval_final_distance_mean": float(np.mean(final_distances)),
    }


def train(n_episodes=None, resume=True):
    if n_episodes is None:
        n_episodes = config_gaussian.TRAIN_EPISODES

    print("Starting training on Gaussian 2D environment...")
    print(f"Bias enabled: {config_gaussian.ENABLE_BIAS}")
    os.makedirs(config_gaussian.RESULTS_DIR, exist_ok=True)
    os.makedirs(config_gaussian.PLOTS_DIR, exist_ok=True)
    os.makedirs(config_gaussian.EVAL_PLOTS_DIR, exist_ok=True)

    env = Gaussian2DEnvironment()
    plot_initial_potential(env)

    agent = PPOAgent(config_gaussian.STATE_SIZE, config_gaussian.ACTION_SIZE, config_gaussian.SEED)
    best_eval_success = -np.inf

    if resume and os.path.exists(config_gaussian.CHECKPOINT_PATH):
        print(f"Loading checkpoint from {config_gaussian.CHECKPOINT_PATH}...")
        ckpt = torch.load(config_gaussian.CHECKPOINT_PATH, map_location="cpu")
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            agent.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            agent.scheduler.load_state_dict(ckpt["scheduler"])
        if "obs_rms" in ckpt:
            agent.load_obs_norm_state(ckpt["obs_rms"])
        if "best_eval_success" in ckpt:
            best_eval_success = float(ckpt["best_eval_success"])

    history = []

    for ep in range(1, n_episodes + 1):
        carry = (
            config_gaussian.CARRY_STATE_ACROSS_EPISODES
            and env.dist_to_target > float(config_gaussian.TARGET_RADIUS)
            and (np.random.rand() > config_gaussian.PROB_FRESH_START)
        )
        state = env.reset(carry_state=carry, episode_index=ep)
        done = False
        steps = 0
        score = 0.0

        while not done and steps < config_gaussian.MAX_ACTIONS_PER_EPISODE:
            action, logp, value = agent.act(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.save_experience(state, action, logp, value, reward, done, next_state)
            state = next_state
            score += reward
            steps += 1

        metrics = agent.update()
        record = {
            "episode": ep,
            "score": float(score),
            "score_per_step": float(score / max(steps, 1)),
            "steps": int(steps),
            "success": float(done),
            "final_distance": float(env.dist_to_target),
            "n_biases": int(len(env.all_biases_in_episode)),
            "carry_state": float(carry),
            "exploration_noise": float(agent.exploration_noise),
            "loss": float(metrics.get("loss", np.nan)),
            "actor_loss": float(metrics.get("actor_loss", np.nan)),
            "critic_loss": float(metrics.get("critic_loss", np.nan)),
            "entropy": float(metrics.get("entropy", np.nan)),
            "approx_kl": float(metrics.get("approx_kl", np.nan)),
            "clip_frac": float(metrics.get("clip_frac", np.nan)),
            "lr": float(metrics.get("lr", np.nan)),
        }

        if ep % config_gaussian.EVAL_EVERY == 0:
            rng_state = np.random.get_state()
            eval_metrics = evaluate(
                agent,
                n_eps=config_gaussian.N_EVAL_EPISODES,
                train_ep=ep,
                seed_base=config_gaussian.SEED,
            )
            np.random.set_state(rng_state)
            record.update(eval_metrics)
            if eval_metrics["eval_success_rate"] >= best_eval_success:
                best_eval_success = eval_metrics["eval_success_rate"]
                save_checkpoint(
                    agent,
                    config_gaussian.BEST_CHECKPOINT_PATH,
                    extra={"episode": ep, "best_eval_success": best_eval_success},
                )
            print(f"EVAL @ {ep}: {eval_metrics}")

        record["best_eval_success"] = float(best_eval_success) if np.isfinite(best_eval_success) else np.nan
        history.append(record)
        write_metrics_csv(history, config_gaussian.METRICS_CSV)

        if ep % config_gaussian.SAVE_TRAJECTORY_EVERY == 0:
            plot_2d_trajectory(env, ep)
        if (ep == 1) or (ep % config_gaussian.SAVE_PLOT_EVERY == 0):
            plot_training_dashboard(history)
            plot_success_analysis(history)

        should_checkpoint = (ep % config_gaussian.SAVE_CHECKPOINT_EVERY == 0) or (ep == n_episodes)
        if should_checkpoint:
            save_checkpoint(
                agent,
                config_gaussian.CHECKPOINT_PATH,
                extra={"episode": ep, "best_eval_success": best_eval_success},
            )

        if ep == 1 or ep % 10 == 0:
            print(
                f"Episode {ep}/{n_episodes} | score={score:.2f} | steps={steps} "
                f"| final_dist={env.dist_to_target:.3f} | success={int(done)} "
                f"| loss={metrics.get('loss', float('nan')):.3f}"
            )

    plot_training_dashboard(history)
    plot_success_analysis(history)
    save_checkpoint(
        agent,
        config_gaussian.CHECKPOINT_PATH,
        extra={"episode": n_episodes, "best_eval_success": best_eval_success},
    )
    print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on the analytical Gaussian 2D environment.")
    parser.add_argument("--episodes", type=int, default=config_gaussian.TRAIN_EPISODES, help="Number of training episodes.")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from the main checkpoint if it exists.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start from scratch and ignore existing checkpoints.")
    parser.add_argument("--replot-only", action="store_true", help="Rebuild dashboard plots from the existing metrics CSV without training.")
    parser.add_argument("--disable-bias", dest="enable_bias", action="store_false", help="Disable hill deposition and run only on the background potential.")
    parser.add_argument("--enable-bias", dest="enable_bias", action="store_true", help="Enable hill deposition and adaptive bias forces.")
    parser.set_defaults(resume=True, enable_bias=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.enable_bias is not None:
        config_gaussian.ENABLE_BIAS = bool(args.enable_bias)
    if args.replot_only:
        history = load_history_from_csv(config_gaussian.METRICS_CSV)
        plot_training_dashboard(history)
        plot_success_analysis(history)
    else:
        train(n_episodes=args.episodes, resume=args.resume)
