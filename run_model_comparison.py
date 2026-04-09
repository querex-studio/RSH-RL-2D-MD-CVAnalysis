from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from comparison_adaptive_cvgen import AdaptiveCVgen2D
from comparison_config import ComparisonConfig, EncoderFitConfig
from comparison_encoders import IdentityEncoder, TICAEncoder, VAMPNetEncoder
from comparison_env import AnalyticalPotentialEnv
from comparison_plots import (
    plot_best_paths,
    plot_comparison_summary,
    plot_potential_landscape,
    plot_training_history,
    write_metrics_csv,
)
from comparison_ppo import PPOAgent


def collect_unbiased_encoder_data(
    config: ComparisonConfig,
    fit_cfg: EncoderFitConfig,
    seed: int,
) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    env_cfg = type(config.env)(**{**config.env.__dict__, "enable_bias": False})
    env = AnalyticalPotentialEnv(env_cfg)
    observations: List[np.ndarray] = []
    steps_used = 0
    for episode_idx in range(fit_cfg.warmup_episodes):
        random_start = episode_idx >= fit_cfg.warmup_episodes // 2
        start_position = None
        if random_start:
            start_position = [
                rng.uniform(env_cfg.domain_min, env_cfg.domain_max),
                rng.uniform(env_cfg.domain_min, env_cfg.domain_max),
            ]
        state = env.reset(start_position=start_position, add_noise=not random_start)
        observations.append(state)
        for _ in range(fit_cfg.warmup_actions_per_episode):
            state, _, done, _ = env.step(0)
            observations.append(state)
            steps_used += env_cfg.sim_steps
            if done:
                state = env.reset(start_position=start_position, add_noise=False)
                observations.append(state)
    return np.asarray(observations, dtype=np.float32), int(steps_used)


def estimate_model_config_rows(config: ComparisonConfig, selected_models: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ppo_steps_per_episode = config.ppo.max_actions_per_episode * config.env.sim_steps
    adaptive_steps_per_round = config.adaptive.replicas * config.adaptive.actions_per_segment * config.env.sim_steps
    warmup_steps = config.encoder_fit.warmup_episodes * config.encoder_fit.warmup_actions_per_episode * config.env.sim_steps
    for model_name in selected_models:
        if model_name == "ppo_biased":
            train_steps = config.total_env_steps_budget
            rows.append(
                {
                    "model": model_name,
                    "controller": "PPO",
                    "encoder": "identity",
                    "bias_enabled": True,
                    "budget_env_steps": config.total_env_steps_budget,
                    "warmup_env_steps": 0,
                    "train_env_steps": train_steps,
                    "approx_episodes_or_rounds": max(1, train_steps // max(1, ppo_steps_per_episode)),
                    "max_actions_per_episode": config.ppo.max_actions_per_episode,
                    "eval_every": config.ppo.eval_every,
                    "eval_episodes": config.ppo.n_eval_episodes,
                    "lag_or_alpha": "-",
                }
            )
        elif model_name == "ppo_tica_2d":
            train_steps = max(config.total_env_steps_budget - warmup_steps, config.env.sim_steps)
            rows.append(
                {
                    "model": model_name,
                    "controller": "PPO",
                    "encoder": "TICA",
                    "bias_enabled": True,
                    "budget_env_steps": config.total_env_steps_budget,
                    "warmup_env_steps": warmup_steps,
                    "train_env_steps": train_steps,
                    "approx_episodes_or_rounds": max(1, train_steps // max(1, ppo_steps_per_episode)),
                    "max_actions_per_episode": config.ppo.max_actions_per_episode,
                    "eval_every": config.ppo.eval_every,
                    "eval_episodes": config.ppo.n_eval_episodes,
                    "lag_or_alpha": f"lag={config.encoder_fit.lagtime}, comps={config.encoder_fit.n_components}",
                }
            )
        elif model_name == "ppo_vampnet_2d":
            train_steps = max(config.total_env_steps_budget - warmup_steps, config.env.sim_steps)
            rows.append(
                {
                    "model": model_name,
                    "controller": "PPO",
                    "encoder": "VAMPNet",
                    "bias_enabled": True,
                    "budget_env_steps": config.total_env_steps_budget,
                    "warmup_env_steps": warmup_steps,
                    "train_env_steps": train_steps,
                    "approx_episodes_or_rounds": max(1, train_steps // max(1, ppo_steps_per_episode)),
                    "max_actions_per_episode": config.ppo.max_actions_per_episode,
                    "eval_every": config.ppo.eval_every,
                    "eval_episodes": config.ppo.n_eval_episodes,
                    "lag_or_alpha": (
                        f"lag={config.vampnet.lagtime}, comps={config.vampnet.n_components}, "
                        f"epochs={config.vampnet.epochs}"
                    ),
                }
            )
        elif model_name == "adaptive_cvgen_like_2d":
            rows.append(
                {
                    "model": model_name,
                    "controller": "Adaptive-CVgen-like",
                    "encoder": "TICA clustering",
                    "bias_enabled": False,
                    "budget_env_steps": config.total_env_steps_budget,
                    "warmup_env_steps": 0,
                    "train_env_steps": config.total_env_steps_budget,
                    "approx_episodes_or_rounds": max(1, config.total_env_steps_budget // max(1, adaptive_steps_per_round)),
                    "max_actions_per_episode": config.adaptive.actions_per_segment,
                    "eval_every": "-",
                    "eval_episodes": "-",
                    "lag_or_alpha": (
                        f"tica_lag={config.adaptive.tica_lagtime}, "
                        f"alpha_candidates={list(config.adaptive.alpha_candidates)}"
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return rows


def write_config_overview_markdown(out_path: Path, rows: Sequence[Mapping[str, object]], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "model",
        "controller",
        "encoder",
        "bias_enabled",
        "budget_env_steps",
        "warmup_env_steps",
        "train_env_steps",
        "approx_episodes_or_rounds",
        "max_actions_per_episode",
        "eval_every",
        "eval_episodes",
        "lag_or_alpha",
    ]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_overview_markdown(
    out_path: Path,
    config_rows: Sequence[Mapping[str, object]],
    result_rows: Sequence[Mapping[str, object]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_map = {str(row["model"]): row for row in result_rows}
    headers = [
        "model",
        "controller",
        "encoder",
        "budget_env_steps",
        "warmup_env_steps",
        "train_env_steps",
        "approx_episodes_or_rounds",
        "success_rate",
        "best_final_distance",
        "n_successes",
    ]
    lines = [
        "# Final Config And Results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for cfg_row in config_rows:
        model_name = str(cfg_row["model"])
        res_row = result_map.get(model_name, {})
        merged = dict(cfg_row)
        merged.update(res_row)
        lines.append("| " + " | ".join(str(merged.get(header, "")) for header in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_effective_config_json(out_path: Path, payload: Mapping[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def evaluate_ppo_agent(agent: PPOAgent, config: ComparisonConfig, n_episodes: int, seed: int):
    env = AnalyticalPotentialEnv(config.env)
    scores = []
    successes = 0
    final_distances = []
    successful_paths = []
    for idx in range(n_episodes):
        np.random.seed(seed + idx)
        raw_state = env.reset()
        score = 0.0
        done = False
        for _ in range(config.ppo.max_actions_per_episode):
            action, _, _ = agent.act(raw_state, training=False)
            raw_state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        artifacts = env.episode_artifacts()
        scores.append(score)
        final_distances.append(artifacts.final_distance)
        if artifacts.success:
            successes += 1
            successful_paths.append(
                {
                    "score": float(score),
                    "path_xy": artifacts.path_xy,
                    "biases": artifacts.biases,
                    "source": "eval",
                }
            )
    return {
        "eval_score_mean": float(np.mean(scores)) if scores else 0.0,
        "eval_success_rate": float(successes / max(1, n_episodes)),
        "eval_final_distance_mean": float(np.mean(final_distances)) if final_distances else float("nan"),
        "successful_paths": successful_paths,
    }


def train_ppo_model(
    model_name: str,
    config: ComparisonConfig,
    encoder,
    output_dir: Path,
    seed: int,
    remaining_env_steps: int,
) -> Dict[str, object]:
    env = AnalyticalPotentialEnv(config.env)
    agent = PPOAgent(config.env, config.ppo, encoder, seed)
    steps_per_episode = config.ppo.max_actions_per_episode * config.env.sim_steps
    n_episodes = max(1, remaining_env_steps // max(1, steps_per_episode))
    history = []
    train_successful_paths = []
    eval_successful_paths = []
    best_biases = []
    best_score = -np.inf
    train_success_counter = 0
    eval_success_counter = 0

    for episode in range(1, n_episodes + 1):
        raw_state = env.reset()
        score = 0.0
        done = False
        actions_taken = 0
        while not done and actions_taken < config.ppo.max_actions_per_episode:
            action, log_prob, value = agent.act(raw_state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(raw_state, action, log_prob, value, reward, done, next_state)
            raw_state = next_state
            score += reward
            actions_taken += 1
        metrics = agent.update()
        artifacts = env.episode_artifacts()
        record = {
            "episode": int(episode),
            "score": float(score),
            "steps": int(actions_taken),
            "success": float(artifacts.success),
            "final_distance": float(artifacts.final_distance),
            "n_biases": int(len(artifacts.biases)),
            "loss": float(metrics.get("loss", np.nan)),
            "actor_loss": float(metrics.get("actor_loss", np.nan)),
            "critic_loss": float(metrics.get("critic_loss", np.nan)),
            "entropy": float(metrics.get("entropy", np.nan)),
            "approx_kl": float(metrics.get("approx_kl", np.nan)),
            "clip_frac": float(metrics.get("clip_frac", np.nan)),
            "lr": float(metrics.get("lr", np.nan)),
        }
        if episode % config.ppo.eval_every == 0 or episode == n_episodes:
            eval_metrics = evaluate_ppo_agent(agent, config, config.ppo.n_eval_episodes, seed + 10_000 + episode)
            record.update(
                {
                    "eval_score_mean": float(eval_metrics["eval_score_mean"]),
                    "eval_success_rate": float(eval_metrics["eval_success_rate"]),
                    "eval_final_distance_mean": float(eval_metrics["eval_final_distance_mean"]),
                }
            )
            for item in eval_metrics["successful_paths"]:
                eval_success_counter += 1
                eval_successful_paths.append(
                    {
                        **item,
                        "label": f"Eval success {eval_success_counter}",
                    }
                )
        history.append(record)
        if artifacts.success:
            train_success_counter += 1
            train_successful_paths.append(
                {
                    "score": float(score),
                    "path_xy": artifacts.path_xy,
                    "biases": artifacts.biases,
                    "source": "train",
                    "label": f"Train success {train_success_counter}",
                }
            )
            if score > best_score:
                best_score = float(score)
                best_biases = list(artifacts.biases)

    train_successful_paths.sort(key=lambda item: item["score"], reverse=True)
    eval_successful_paths.sort(key=lambda item: item["score"], reverse=True)
    top_train_paths = train_successful_paths[:2]
    top_eval_paths = eval_successful_paths[:2]
    if top_train_paths and top_eval_paths:
        top_paths = [top_train_paths[0], top_eval_paths[0]]
    else:
        combined = sorted(train_successful_paths + eval_successful_paths, key=lambda item: item["score"], reverse=True)
        top_paths = combined[:2]
    write_metrics_csv(history, output_dir / "metrics.csv")
    plot_training_history(history, output_dir / "training_history.png", model_name)
    plot_potential_landscape(env, output_dir / "unbiased_potential.png", biases=None, title=f"{model_name} Unbiased Potential")
    plot_potential_landscape(
        env,
        output_dir / "biased_potential.png",
        biases=best_biases if best_biases else None,
        title=f"{model_name} Biased Potential",
    )
    plot_best_paths(
        env,
        top_paths,
        output_dir / "best_two_paths.png",
        biases=best_biases if best_biases else None,
        title=f"{model_name} Best Successful Paths",
    )
    plot_best_paths(
        env,
        top_train_paths,
        output_dir / "best_train_two_paths.png",
        biases=best_biases if best_biases else None,
        title=f"{model_name} Best Train Successful Paths",
    )
    plot_best_paths(
        env,
        top_eval_paths,
        output_dir / "best_eval_two_paths.png",
        biases=best_biases if best_biases else None,
        title=f"{model_name} Best Eval Successful Paths",
    )
    summary = {
        "model": model_name,
        "history": history,
        "success_rate": float(np.mean([row["success"] for row in history])) if history else 0.0,
        "best_final_distance": float(np.min([row["final_distance"] for row in history])) if history else float("nan"),
        "n_successes": int(sum(1 for row in history if float(row["success"]) > 0.5)),
        "top_paths": top_paths,
        "top_train_paths": top_train_paths,
        "top_eval_paths": top_eval_paths,
    }
    return summary


def run_adaptive_cvgen_model(model_name: str, config: ComparisonConfig, output_dir: Path, seed: int) -> Dict[str, object]:
    runner = AdaptiveCVgen2D(config.env, config.adaptive, seed)
    result = runner.run(config.total_env_steps_budget)
    env = AnalyticalPotentialEnv(type(config.env)(**{**config.env.__dict__, "enable_bias": False}))
    successful_paths = [
        {
            "score": float(item.reward_score),
            "path_xy": item.full_path,
            "biases": [],
            "source": "train",
            "label": f"Adaptive success {idx + 1}",
        }
        for idx, item in enumerate(result["successful_paths"])
    ]
    history = result["history"]
    write_metrics_csv(history, output_dir / "metrics.csv")
    plot_training_history(history, output_dir / "training_history.png", model_name)
    plot_potential_landscape(env, output_dir / "unbiased_potential.png", biases=None, title=f"{model_name} Unbiased Potential")
    plot_potential_landscape(env, output_dir / "biased_potential.png", biases=None, title=f"{model_name} Biased Potential (None)")
    plot_best_paths(env, successful_paths, output_dir / "best_two_paths.png", biases=None, title=f"{model_name} Best Successful Paths")
    plot_best_paths(env, successful_paths[:2], output_dir / "best_train_two_paths.png", biases=None, title=f"{model_name} Best Train Successful Paths")
    plot_best_paths(env, [], output_dir / "best_eval_two_paths.png", biases=None, title=f"{model_name} Best Eval Successful Paths")
    summary = {
        "model": model_name,
        "history": history,
        "success_rate": float(min(1.0, len(successful_paths) / max(1, len(history)))) if history else 0.0,
        "best_final_distance": float(np.min([row["best_distance"] for row in history])) if history else float("nan"),
        "n_successes": int(len(successful_paths)),
        "top_paths": successful_paths[:2],
        "top_train_paths": successful_paths[:2],
        "top_eval_paths": [],
    }
    return summary


def save_summary(output_dir: Path, summary: Dict[str, object]) -> None:
    serializable = dict(summary)
    serializable.pop("history", None)
    for key in ("top_paths", "top_train_paths", "top_eval_paths"):
        compact_paths = []
        for item in serializable.get(key, []):
            compact_paths.append(
                {
                    "label": item["label"],
                    "score": float(item["score"]),
                    "source": item.get("source", ""),
                    "n_points": int(len(item["path_xy"])),
                }
            )
        serializable[key] = compact_paths
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the four-model analytical-potential comparison.")
    parser.add_argument(
        "--models",
        type=str,
        default="ppo_biased,adaptive_cvgen_like_2d,ppo_tica_2d,ppo_vampnet_2d",
        help="Comma-separated model names.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a very small verification budget.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ComparisonConfig()
    if args.smoke_test:
        config.total_env_steps_budget = 4_000
        config.encoder_fit.warmup_episodes = 8
        config.encoder_fit.warmup_actions_per_episode = 8
        config.ppo.max_actions_per_episode = 12
        config.ppo.eval_every = 5
        config.ppo.n_eval_episodes = 3
        config.adaptive.replicas = 4
        config.adaptive.actions_per_segment = 4
        config.vampnet.epochs = 8
        config.vampnet.batch_size = 64

    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    output_root = config.output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    model_summaries = []
    config_rows = estimate_model_config_rows(config, selected_models)
    write_config_overview_markdown(output_root / "config_overview_start.md", config_rows, "Model Config Overview")
    model_seeds = {
        "ppo_biased": config.seed,
        "adaptive_cvgen_like_2d": config.seed,
        "ppo_tica_2d": config.seed,
        "ppo_vampnet_2d": config.seed,
    }

    for model_name in selected_models:
        output_dir = config.model_dir(model_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_seed = model_seeds.get(model_name, config.seed)
        effective_row = next(row for row in config_rows if str(row["model"]) == model_name)
        write_model_effective_config_json(output_dir / "effective_config.json", {"seed": model_seed, **effective_row})
        if model_name == "ppo_biased":
            encoder = IdentityEncoder(config.env.state_size)
            encoder.fit(np.zeros((2, config.env.state_size), dtype=np.float32))
            summary = train_ppo_model(model_name, config, encoder, output_dir, model_seed, config.total_env_steps_budget)
        elif model_name == "ppo_tica_2d":
            fit_cfg = config.encoder_fit
            warmup_obs, warmup_steps = collect_unbiased_encoder_data(config, fit_cfg, model_seed + 100)
            encoder = TICAEncoder(fit_cfg)
            encoder.fit(warmup_obs)
            summary = train_ppo_model(
                model_name,
                config,
                encoder,
                output_dir,
                model_seed,
                max(config.total_env_steps_budget - warmup_steps, config.env.sim_steps),
            )
            summary["encoder_warmup_steps"] = int(warmup_steps)
        elif model_name == "ppo_vampnet_2d":
            fit_cfg = config.encoder_fit
            warmup_obs, warmup_steps = collect_unbiased_encoder_data(config, fit_cfg, model_seed + 200)
            encoder = VAMPNetEncoder(fit_cfg, config.vampnet)
            encoder.fit(warmup_obs)
            summary = train_ppo_model(
                model_name,
                config,
                encoder,
                output_dir,
                model_seed,
                max(config.total_env_steps_budget - warmup_steps, config.env.sim_steps),
            )
            summary["encoder_warmup_steps"] = int(warmup_steps)
        elif model_name == "adaptive_cvgen_like_2d":
            summary = run_adaptive_cvgen_model(model_name, config, output_dir, model_seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        save_summary(output_dir, summary)
        model_summaries.append(summary)

    plot_comparison_summary(model_summaries, output_root / "comparison_summary.png")
    with (output_root / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        compact = []
        for item in model_summaries:
            compact.append(
                {
                    "model": item["model"],
                    "success_rate": float(item["success_rate"]),
                    "best_final_distance": float(item["best_final_distance"]),
                    "n_successes": int(item["n_successes"]),
                }
            )
        json.dump(compact, handle, indent=2)
    write_results_overview_markdown(output_root / "config_and_results_final.md", config_rows, compact)


if __name__ == "__main__":
    main()
