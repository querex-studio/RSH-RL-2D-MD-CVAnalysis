from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from comparison_env import AnalyticalPotentialEnv


def write_metrics_csv(history: Sequence[Mapping[str, object]], out_path: Path) -> None:
    if not history:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def make_landscape_grid(env: AnalyticalPotentialEnv):
    cfg = env.cfg
    xs = np.linspace(cfg.domain_min, cfg.domain_max, cfg.plot_grid_size)
    ys = np.linspace(cfg.domain_min, cfg.domain_max, cfg.plot_grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)
    background = env.background_potential_grid(x_grid, y_grid)
    return x_grid, y_grid, background


def plot_potential_landscape(
    env: AnalyticalPotentialEnv,
    out_path: Path,
    biases: Optional[Iterable[tuple[float, float, float, float]]] = None,
    title: str = "Potential Landscape",
) -> None:
    x_grid, y_grid, background = make_landscape_grid(env)
    bias_grid = env.bias_potential_grid(x_grid, y_grid, biases=biases)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    lo, hi = np.percentile(background, [2, 98])
    if hi <= lo:
        lo = float(background.min())
        hi = float(background.max()) + 1e-6
    cf = ax.contourf(x_grid, y_grid, background, levels=80, cmap="winter", norm=Normalize(vmin=lo, vmax=hi))
    ax.contour(x_grid, y_grid, background, levels=22, colors="k", alpha=0.55, linewidths=0.8)

    divider = make_axes_locatable(ax)
    cax_bias = divider.append_axes("right", size="4.5%", pad=0.45)
    cax_psp = divider.append_axes("right", size="4.5%", pad=0.45)
    fig.subplots_adjust(right=0.88)

    if np.any(bias_grid > 0):
        vmax_b = np.percentile(bias_grid[bias_grid > 0], 98)
        bias_im = ax.imshow(
            bias_grid,
            extent=[env.cfg.domain_min, env.cfg.domain_max, env.cfg.domain_min, env.cfg.domain_max],
            origin="lower",
            cmap="OrRd",
            alpha=0.6,
            norm=PowerNorm(gamma=0.9, vmin=0.0, vmax=max(vmax_b, 1e-6), clip=True),
            interpolation="none",
            zorder=2,
        )
        cb_bias = fig.colorbar(bias_im, cax=cax_bias)
        cb_bias.set_label("Bias")
    else:
        cax_bias.remove()

    cb_psp = fig.colorbar(cf, cax=cax_psp)
    cb_psp.set_label("Potential")
    ax.plot(env.cfg.start_x, env.cfg.start_y, "go", markersize=8, label="Start")
    ax.plot(env.cfg.target_x, env.cfg.target_y, "r*", markersize=11, label="Target")
    ax.add_patch(
        plt.Circle(
            (env.cfg.target_x, env.cfg.target_y),
            env.cfg.target_radius,
            color="r",
            fill=False,
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax.set_xlim(env.cfg.domain_min, env.cfg.domain_max)
    ax.set_ylim(env.cfg.domain_min, env.cfg.domain_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_paths(
    env: AnalyticalPotentialEnv,
    paths: Sequence[Mapping[str, object]],
    out_path: Path,
    biases: Optional[Iterable[tuple[float, float, float, float]]] = None,
    title: str = "Best Successful Transition Paths",
) -> None:
    x_grid, y_grid, background = make_landscape_grid(env)
    bias_grid = env.bias_potential_grid(x_grid, y_grid, biases=biases)
    fig, ax = plt.subplots(figsize=(7.3, 6.3))
    lo, hi = np.percentile(background, [2, 98])
    cf = ax.contourf(x_grid, y_grid, background, levels=80, cmap="winter", norm=Normalize(vmin=lo, vmax=max(hi, lo + 1e-6)))
    ax.contour(x_grid, y_grid, background, levels=22, colors="k", alpha=0.4, linewidths=0.8)
    if np.any(bias_grid > 0):
        vmax_b = np.percentile(bias_grid[bias_grid > 0], 98)
        ax.imshow(
            bias_grid,
            extent=[env.cfg.domain_min, env.cfg.domain_max, env.cfg.domain_min, env.cfg.domain_max],
            origin="lower",
            cmap="OrRd",
            alpha=0.45,
            norm=PowerNorm(gamma=0.9, vmin=0.0, vmax=max(vmax_b, 1e-6), clip=True),
            interpolation="none",
            zorder=2,
        )
    colors = ["black", "gold"]
    for idx, path_info in enumerate(paths[:2]):
        path_xy = np.asarray(path_info["path_xy"], dtype=float)
        if len(path_xy) == 0:
            continue
        ax.plot(path_xy[:, 0], path_xy[:, 1], color=colors[idx], linewidth=2.2, alpha=0.95, label=path_info["label"])
        ax.plot(path_xy[0, 0], path_xy[0, 1], marker="o", color=colors[idx], markersize=6)
        ax.plot(path_xy[-1, 0], path_xy[-1, 1], marker="X", color=colors[idx], markersize=9)
    ax.plot(env.cfg.start_x, env.cfg.start_y, "go", markersize=8, label="Start")
    ax.plot(env.cfg.target_x, env.cfg.target_y, "r*", markersize=11, label="Target")
    ax.add_patch(
        plt.Circle(
            (env.cfg.target_x, env.cfg.target_y),
            env.cfg.target_radius,
            color="r",
            fill=False,
            linestyle="--",
            linewidth=1.2,
        )
    )
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label("Potential")
    ax.set_xlim(env.cfg.domain_min, env.cfg.domain_max)
    ax.set_ylim(env.cfg.domain_min, env.cfg.domain_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=env.cfg.trajectory_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: Sequence[Mapping[str, object]], out_path: Path, title_prefix: str) -> None:
    if not history:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "episode" in history[0]:
        x = np.array([row["episode"] for row in history], dtype=float)
        score = np.array([row["score"] for row in history], dtype=float)
        success = np.array([row["success"] for row in history], dtype=float)
        final_distance = np.array([row["final_distance"] for row in history], dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
        axes[0].plot(x, score, color="steelblue")
        axes[0].set_title(f"{title_prefix}: Reward")
        axes[1].plot(x, success, color="darkgreen")
        axes[1].set_title(f"{title_prefix}: Success")
        axes[1].set_ylim(-0.05, 1.05)
        axes[2].plot(x, final_distance, color="firebrick")
        axes[2].set_title(f"{title_prefix}: Final Distance")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.set_xlabel("Episode")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    x = np.array([row["round"] for row in history], dtype=float)
    best_distance = np.array([row["best_distance"] for row in history], dtype=float)
    coverage = np.array([row["coverage_ratio"] for row in history], dtype=float)
    successes = np.array([row["cumulative_successes"] for row in history], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axes[0].plot(x, best_distance, color="firebrick")
    axes[0].set_title(f"{title_prefix}: Best Distance")
    axes[1].plot(x, coverage, color="purple")
    axes[1].set_title(f"{title_prefix}: Coverage")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].plot(x, successes, color="darkgreen")
    axes[2].set_title(f"{title_prefix}: Cumulative Successes")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlabel("Round")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_summary(model_summaries: Sequence[Mapping[str, object]], out_path: Path) -> None:
    if not model_summaries:
        return
    labels = [str(item["model"]) for item in model_summaries]
    success = np.array([float(item.get("success_rate", 0.0)) for item in model_summaries], dtype=float)
    final_distance = np.array([float(item.get("best_final_distance", np.nan)) for item in model_summaries], dtype=float)
    n_successes = np.array([float(item.get("n_successes", 0.0)) for item in model_summaries], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axes[0].bar(labels, success, color=["steelblue", "teal", "darkorange", "purple"][: len(labels)])
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].bar(labels, final_distance, color=["firebrick", "olive", "goldenrod", "slateblue"][: len(labels)])
    axes[1].set_title("Best Final Distance")
    axes[2].bar(labels, n_successes, color=["darkgreen", "seagreen", "darkcyan", "navy"][: len(labels)])
    axes[2].set_title("Successful Paths")
    for ax in axes:
        ax.grid(alpha=0.2, axis="y")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
