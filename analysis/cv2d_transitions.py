import argparse
import json
import os
import sys
from typing import Tuple, Optional, Dict, List

import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils

KB_KCAL = 0.00198720425864083  # kcal/mol/K
KB_KJ = 0.00831446261815324    # kJ/mol/K


def find_latest_run(root):
    if not os.path.isdir(root):
        return None
    candidates = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def load_cv_from_run(run_dir) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    cv1_path = os.path.join(run_dir, "data", "cv1.npy")
    cv2_path = os.path.join(run_dir, "data", "cv2.npy")
    if not (os.path.exists(cv1_path) and os.path.exists(cv2_path)):
        raise FileNotFoundError("cv1.npy / cv2.npy not found in run_dir/data.")
    cv1 = np.load(cv1_path)
    cv2 = np.load(cv2_path)

    time = None
    for name in ("time.npy", "t.npy", "times.npy"):
        t_path = os.path.join(run_dir, "data", name)
        if os.path.exists(t_path):
            time = np.load(t_path)
            break

    episode = None
    for name in ("episode.npy", "episodes.npy", "ep.npy"):
        e_path = os.path.join(run_dir, "data", name)
        if os.path.exists(e_path):
            episode = np.load(e_path)
            break

    return cv1, cv2, time, episode


def load_cv_from_csv(path, cv1_col=None, cv2_col=None, time_col=None, episode_col=None):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.dtype.names:
        names = list(arr.dtype.names)

        def _pick(col_name_candidates):
            for name in col_name_candidates:
                if name in names:
                    return arr[name]
            return None

        time = _pick(["time", "t", "step", "frame", "frames"])
        episode = _pick(["episode", "ep", "episode_id", "traj"])
        cv1 = _pick(["cv1", "cv1_A", "cv1_distance", "cv1_distance_A"])
        cv2 = _pick(["cv2", "cv2_A", "cv2_distance", "cv2_distance_A"])
        if cv1 is not None and cv2 is not None:
            return np.asarray(cv1), np.asarray(cv2), np.asarray(time) if time is not None else None, np.asarray(episode) if episode is not None else None

    raw = np.genfromtxt(path, delimiter=",")
    if raw.ndim == 1:
        raise ValueError("CSV appears to have a single row/column; cannot infer cv1/cv2.")
    if cv1_col is None or cv2_col is None:
        cv1_col = 1 if cv1_col is None else cv1_col
        cv2_col = 2 if cv2_col is None else cv2_col
    time = raw[:, time_col] if time_col is not None else (raw[:, 0] if raw.shape[1] >= 3 else None)
    episode = raw[:, episode_col] if episode_col is not None else None
    return raw[:, cv1_col], raw[:, cv2_col], time, episode


def freedman_diaconis_bins(x, min_bins=30, max_bins=200):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return min_bins
    bin_width = 2.0 * iqr * (len(x) ** (-1.0 / 3.0))
    if bin_width <= 0:
        return min_bins
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return int(np.clip(bins, min_bins, max_bins))


def scott_bins(x, min_bins=30, max_bins=200):
    sigma = np.std(x)
    if sigma <= 0:
        return min_bins
    bin_width = 3.5 * sigma * (len(x) ** (-1.0 / 3.0))
    if bin_width <= 0:
        return min_bins
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return int(np.clip(bins, min_bins, max_bins))


def sturges_bins(x, min_bins=30, max_bins=200):
    bins = int(np.ceil(np.log2(len(x)) + 1))
    return int(np.clip(bins, min_bins, max_bins))


def sqrt_bins(x, min_bins=30, max_bins=200):
    bins = int(np.ceil(np.sqrt(len(x))))
    return int(np.clip(bins, min_bins, max_bins))


def compute_bins(x, y, bins=None, bin_width=None, min_bins=30, max_bins=200, bin_strategy="auto"):
    if bins is not None:
        if isinstance(bins, (list, tuple)) and len(bins) == 2:
            return bins[0], bins[1]
        return int(bins), int(bins)
    if bin_width is not None:
        if isinstance(bin_width, (list, tuple)) and len(bin_width) == 2:
            bwx, bwy = bin_width
        else:
            bwx = bwy = float(bin_width)
        nx = int(np.ceil((x.max() - x.min()) / bwx))
        ny = int(np.ceil((y.max() - y.min()) / bwy))
        return int(np.clip(nx, min_bins, max_bins)), int(np.clip(ny, min_bins, max_bins))
    strategy = (bin_strategy or "auto").lower()
    if strategy == "fd":
        return freedman_diaconis_bins(x, min_bins, max_bins), freedman_diaconis_bins(y, min_bins, max_bins)
    if strategy == "scott":
        return scott_bins(x, min_bins, max_bins), scott_bins(y, min_bins, max_bins)
    if strategy == "sturges":
        return sturges_bins(x, min_bins, max_bins), sturges_bins(y, min_bins, max_bins)
    if strategy == "sqrt":
        return sqrt_bins(x, min_bins, max_bins), sqrt_bins(y, min_bins, max_bins)

    bx = freedman_diaconis_bins(x, min_bins, max_bins)
    by = freedman_diaconis_bins(y, min_bins, max_bins)
    if bx == min_bins and np.std(x) > 0:
        bx = scott_bins(x, min_bins, max_bins)
    if by == min_bins and np.std(y) > 0:
        by = scott_bins(y, min_bins, max_bins)
    if bx == min_bins and np.std(x) == 0:
        bx = sturges_bins(x, min_bins, max_bins)
    if by == min_bins and np.std(y) == 0:
        by = sturges_bins(y, min_bins, max_bins)
    return bx, by


def assign_basins_manual(cv1, cv2, basin_rects, basin_polys):
    n = len(cv1)
    labels = np.full(n, -1, dtype=int)
    names: Dict[int, str] = {}
    idx = 0

    for rect in basin_rects:
        name = rect["name"]
        xmin, xmax, ymin, ymax = rect["xmin"], rect["xmax"], rect["ymin"], rect["ymax"]
        mask = (cv1 >= xmin) & (cv1 <= xmax) & (cv2 >= ymin) & (cv2 <= ymax)
        assign = (labels == -1) & mask
        labels[assign] = idx
        names[idx] = name
        idx += 1

    for poly in basin_polys:
        name = poly["name"]
        points = np.asarray(poly["points"], dtype=float)
        path = Path(points)
        mask = path.contains_points(np.column_stack([cv1, cv2]))
        assign = (labels == -1) & mask
        labels[assign] = idx
        names[idx] = name
        idx += 1

    return labels, names


def _zscore(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def assign_basins_auto(
    cv1,
    cv2,
    method,
    per_episode,
    episodes,
    scale,
    dbscan_eps,
    dbscan_min_samples,
    gmm_components,
    gmm_cov,
    gmm_prob_threshold,
):
    try:
        if method == "dbscan":
            from sklearn.cluster import DBSCAN
        else:
            from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise ImportError("scikit-learn is required for auto basin clustering.") from exc

    n = len(cv1)
    labels = np.full(n, -1, dtype=int)
    names: Dict[int, str] = {}
    label_offset = 0

    if episodes is None:
        episodes = np.zeros(n, dtype=int)
    unique_eps = list(dict.fromkeys(episodes.tolist()))
    if not per_episode:
        unique_eps = [unique_eps[0]]
        ep_mask = np.ones(n, dtype=bool)
        ep_indices = [np.where(ep_mask)[0]]
        ep_values = [0]
    else:
        ep_indices = [np.where(episodes == ep)[0] for ep in unique_eps]
        ep_values = unique_eps

    for ep_val, idxs in zip(ep_values, ep_indices):
        if idxs.size == 0:
            continue
        X = np.column_stack([cv1[idxs], cv2[idxs]])
        if scale == "zscore":
            X = _zscore(X)

        if method == "dbscan":
            model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            lab = model.fit_predict(X)
            for c in sorted(set(lab)):
                if c == -1:
                    continue
                names[label_offset + c] = f"E{ep_val}_C{c}" if per_episode else f"C{c}"
            mapped = np.full(lab.shape, -1, dtype=int)
            valid = lab >= 0
            mapped[valid] = lab[valid] + label_offset
            labels[idxs] = mapped
            if lab.size > 0 and np.max(lab) >= 0:
                label_offset += int(np.max(lab)) + 1
        else:
            model = GaussianMixture(n_components=gmm_components, covariance_type=gmm_cov, random_state=0)
            model.fit(X)
            prob = model.predict_proba(X)
            lab = np.argmax(prob, axis=1)
            maxp = np.max(prob, axis=1)
            mapped = np.full(lab.shape, -1, dtype=int)
            valid = maxp >= gmm_prob_threshold
            mapped[valid] = lab[valid] + label_offset
            labels[idxs] = mapped
            for c in range(gmm_components):
                names[label_offset + c] = f"E{ep_val}_C{c}" if per_episode else f"C{c}"
            label_offset += gmm_components

    return labels, names


def extract_transition_segments(labels, episodes, transition_mode="outside", boundary_pad=0):
    if episodes is None:
        episodes = np.zeros_like(labels, dtype=int)
    unique_eps = list(dict.fromkeys(episodes.tolist()))
    segments = []
    transition_indices = []
    seg_id = 0

    for ep in unique_eps:
        idxs = np.where(episodes == ep)[0]
        if idxs.size == 0:
            continue
        lab = labels[idxs]

        if transition_mode == "boundary":
            for i in range(1, len(lab)):
                if lab[i] == -1 or lab[i - 1] == -1:
                    continue
                if lab[i] != lab[i - 1]:
                    start = max(0, i - 1 - boundary_pad)
                    end = min(len(lab) - 1, i + boundary_pad)
                    seg_id += 1
                    seg_indices = idxs[start:end + 1]
                    transition_indices.extend(seg_indices.tolist())
                    segments.append({
                        "segment_id": seg_id,
                        "episode": int(ep),
                        "basin_from": int(lab[i - 1]),
                        "basin_to": int(lab[i]),
                        "start_index": int(seg_indices[0]),
                        "end_index": int(seg_indices[-1]),
                    })
            continue

        # transition_mode == "outside"
        i = 0
        while i < len(lab):
            if lab[i] != -1:
                i += 1
                continue
            start = i
            while i < len(lab) and lab[i] == -1:
                i += 1
            end = i - 1
            left = start - 1
            right = end + 1
            basin_from = lab[left] if left >= 0 else -1
            basin_to = lab[right] if right < len(lab) else -1
            if basin_from == -1 or basin_to == -1 or basin_from == basin_to:
                continue
            seg_id += 1
            seg_indices = idxs[start:end + 1]
            transition_indices.extend(seg_indices.tolist())
            segments.append({
                "segment_id": seg_id,
                "episode": int(ep),
                "basin_from": int(basin_from),
                "basin_to": int(basin_to),
                "start_index": int(seg_indices[0]),
                "end_index": int(seg_indices[-1]),
            })

    transition_indices = np.array(sorted(set(transition_indices)), dtype=int)
    return segments, transition_indices


def format_episode_label(ep):
    try:
        val = float(ep)
        if abs(val - int(val)) < 1e-6:
            return f"ep{int(val)}"
        return f"ep{str(val).replace('.', 'p')}"
    except Exception:
        return f"ep{str(ep)}"


def main():
    parser = argparse.ArgumentParser(description="Transition path density and pathway extraction in CV space.")
    parser.add_argument("--run", default=None, help="Run directory with data/cv1.npy, data/cv2.npy.")
    parser.add_argument("--runs-root", default=None, help="Root folder for analysis_runs.")
    parser.add_argument("--cv1", dest="cv1_path", default=None, help="Optional path to cv1.npy.")
    parser.add_argument("--cv2", dest="cv2_path", default=None, help="Optional path to cv2.npy.")
    parser.add_argument("--time", dest="time_path", default=None, help="Optional path to time.npy.")
    parser.add_argument("--episode", dest="episode_path", default=None, help="Optional path to episode.npy.")
    parser.add_argument("--csv", default=None, help="CSV with time, cv1, cv2 columns.")
    parser.add_argument("--cv1-col", type=int, default=None, help="CSV column index for CV1 (0-based).")
    parser.add_argument("--cv2-col", type=int, default=None, help="CSV column index for CV2 (0-based).")
    parser.add_argument("--time-col", type=int, default=None, help="CSV column index for time (0-based).")
    parser.add_argument("--episode-col", type=int, default=None, help="CSV column index for episode (0-based).")

    parser.add_argument("--basin-method", choices=["auto", "manual"], default="auto")
    parser.add_argument("--basin-rect", action="append", default=[], help="Rect basin: name,xmin,xmax,ymin,ymax")
    parser.add_argument("--basin-poly", default=None, help="JSON file with polygon basins.")

    parser.add_argument("--cluster-method", choices=["dbscan", "gmm"], default="dbscan")
    parser.add_argument("--cluster-per-episode", action="store_true")
    parser.add_argument("--cluster-scale", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--dbscan-eps", type=float, default=0.2)
    parser.add_argument("--dbscan-min-samples", type=int, default=30)
    parser.add_argument("--gmm-components", type=int, default=4)
    parser.add_argument("--gmm-cov", choices=["full", "tied", "diag", "spherical"], default="full")
    parser.add_argument("--gmm-prob-threshold", type=float, default=0.6)

    parser.add_argument("--transition-mode", choices=["outside", "boundary"], default="outside")
    parser.add_argument("--boundary-pad", type=int, default=0)

    parser.add_argument("--bins", type=int, default=None, help="Number of bins (applied to both axes).")
    parser.add_argument("--bin-width", type=float, default=None, help="Bin width (A) for both axes.")
    parser.add_argument("--min-bins", type=int, default=30)
    parser.add_argument("--max-bins", type=int, default=200)
    parser.add_argument("--bin-strategy", choices=["auto", "fd", "scott", "sturges", "sqrt"], default="auto")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--units", choices=["kcal", "kJ"], default="kcal")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--pseudocount", type=float, default=0.0)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--trans-min-count", type=int, default=2)
    parser.add_argument("--trans-alpha", type=float, default=0.55)
    parser.add_argument("--trans-overlay", choices=["heatmap", "contour"], default="heatmap")

    parser.add_argument("--color-by", choices=["time", "episode"], default="time")
    parser.add_argument("--time-units", default="step")
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--point-alpha", type=float, default=0.8)
    parser.add_argument("--per-episode", action="store_true", help="Also compute per-episode transition outputs.")
    parser.add_argument("--no-pooled", action="store_true", help="Skip pooled (overall) outputs.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes for per-episode outputs.")
    parser.add_argument("--out-prefix", default="cv2d_transition", help="Output filename prefix.")
    parser.add_argument("--config-module", default=None, help="Config module for labels (default: combined_2d or config).")
    parser.add_argument("--cv1-label", default=None, help="Label for CV1 (overrides config label).")
    parser.add_argument("--cv2-label", default=None, help="Label for CV2 (overrides config label).")
    args = parser.parse_args()

    runs_root = args.runs_root or os.path.join(ROOT_DIR, "results_PPO", "analysis_runs")
    run_dir = args.run
    if run_dir is not None and not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT_DIR, run_dir)
    if run_dir is None:
        run_dir = find_latest_run(runs_root)
        if run_dir is None:
            time_tag = run_utils.default_time_tag()
            run_dir = run_utils.prepare_run_dir(time_tag, root=runs_root)
            run_utils.write_run_metadata(run_dir, {"script": "analysis/cv2d_transitions.py"})

    cfg_name = args.config_module or ("combined_2d" if os.path.exists(os.path.join(ROOT_DIR, "combined_2d.py")) else "config")
    try:
        cfg = importlib.import_module(cfg_name)
    except Exception:
        cfg = None

    if args.cv1_path and args.cv2_path:
        cv1 = np.load(args.cv1_path)
        cv2 = np.load(args.cv2_path)
        time = np.load(args.time_path) if args.time_path else None
        episode = np.load(args.episode_path) if args.episode_path else None
    elif args.csv:
        cv1, cv2, time, episode = load_cv_from_csv(args.csv, args.cv1_col, args.cv2_col, args.time_col, args.episode_col)
    else:
        cv1, cv2, time, episode = load_cv_from_run(run_dir)

    cv1 = np.asarray(cv1).reshape(-1)
    cv2 = np.asarray(cv2).reshape(-1)
    if time is not None:
        time = np.asarray(time).reshape(-1)
        if time.size != cv1.size:
            raise ValueError("time must have the same length as cv1/cv2.")
    if episode is not None:
        episode = np.asarray(episode).reshape(-1)
        if episode.size != cv1.size:
            raise ValueError("episode must have the same length as cv1/cv2.")

    if args.basin_method == "manual":
        basin_rects = []
        for entry in args.basin_rect:
            parts = [p.strip() for p in entry.split(",")]
            if len(parts) != 5:
                raise ValueError("basin-rect must be: name,xmin,xmax,ymin,ymax")
            name, xmin, xmax, ymin, ymax = parts
            basin_rects.append({
                "name": name,
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
            })
        basin_polys = []
        if args.basin_poly:
            with open(args.basin_poly, "r", encoding="utf-8") as f:
                poly_data = json.load(f)
            for poly in poly_data:
                basin_polys.append({
                    "name": poly["name"],
                    "points": poly["points"],
                })
        if not basin_rects and not basin_polys:
            raise ValueError("Manual basin method requires --basin-rect or --basin-poly.")
        labels, basin_names = assign_basins_manual(cv1, cv2, basin_rects, basin_polys)
    else:
        labels, basin_names = assign_basins_auto(
            cv1,
            cv2,
            method=args.cluster_method,
            per_episode=args.cluster_per_episode,
            episodes=episode,
            scale=args.cluster_scale,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            gmm_components=args.gmm_components,
            gmm_cov=args.gmm_cov,
            gmm_prob_threshold=args.gmm_prob_threshold,
        )

    segments, trans_idx = extract_transition_segments(
        labels, episode, transition_mode=args.transition_mode, boundary_pad=args.boundary_pad
    )

    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Save transition segments CSV
    trans_csv = os.path.join(data_dir, f"{args.out_prefix}_segments.csv")
    with open(trans_csv, "w", encoding="utf-8") as f:
        f.write("segment_id,episode,time,cv1,cv2,basin_from,basin_to,index\n")
        for seg in segments:
            seg_id = seg["segment_id"]
            ep = seg["episode"]
            basin_from = basin_names.get(seg["basin_from"], str(seg["basin_from"]))
            basin_to = basin_names.get(seg["basin_to"], str(seg["basin_to"]))
            for idx in range(seg["start_index"], seg["end_index"] + 1):
                tval = time[idx] if time is not None else idx
                f.write(f"{seg_id},{ep},{tval},{cv1[idx]},{cv2[idx]},{basin_from},{basin_to},{idx}\n")

    # Save basin map
    basin_meta = {
        "basin_method": args.basin_method,
        "basin_names": {str(k): v for k, v in basin_names.items()},
        "cluster_method": args.cluster_method if args.basin_method == "auto" else None,
        "cluster_per_episode": args.cluster_per_episode,
    }
    with open(os.path.join(data_dir, f"{args.out_prefix}_basins.json"), "w", encoding="utf-8") as f:
        json.dump(basin_meta, f, indent=2)

    label_cv1 = args.cv1_label or (getattr(cfg, "CV1_LABEL", "CV1") if cfg else "CV1")
    label_cv2 = args.cv2_label or (getattr(cfg, "CV2_LABEL", "CV2") if cfg else "CV2")

    def _transition_scatter(indices, suffix, title_suffix=None):
        if indices.size == 0:
            return
        if args.color_by == "episode":
            color_val = episode[indices] if episode is not None else np.zeros_like(indices, dtype=float)
            color_label = "episode"
        else:
            color_val = time[indices] if time is not None else indices.astype(float)
            color_label = f"time ({args.time_units})"

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(cv1[indices], cv2[indices], c=color_val, s=args.point_size,
                         alpha=args.point_alpha, cmap="viridis", linewidths=0)
        plt.colorbar(sc, label=color_label)
        plt.xlabel(f"{label_cv1} (distance, A)")
        plt.ylabel(f"{label_cv2} (distance, A)")
        title = "Transition-Only CV1 vs CV2"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}{suffix}_transition_scatter.png"), dpi=220)
        plt.close()

    def _fes_from_counts(counts):
        if args.pseudocount > 0:
            counts_pc = counts + args.pseudocount
            prob = counts_pc / np.sum(counts_pc)
        else:
            prob = counts / np.sum(counts)
        kB = KB_KCAL if args.units == "kcal" else KB_KJ
        kT = kB * float(args.temperature)
        fes = -kT * np.log(prob + args.epsilon)
        if np.isfinite(fes).any():
            fes = fes - np.nanmin(fes)
        fes[counts < args.min_count] = np.nan
        return fes

    def _transition_overlay(counts_all, counts_trans, xedges, yedges, suffix, title_suffix=None):
        fes = _fes_from_counts(counts_all)
        plt.figure(figsize=(6, 5))
        vmin = np.nanmin(fes) if np.isfinite(fes).any() else None
        vmax = np.nanpercentile(fes[np.isfinite(fes)], 95) if np.isfinite(fes).any() else None
        plt.imshow(fes.T, origin="lower", aspect="auto", cmap="coolwarm",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   vmin=vmin, vmax=vmax)
        plt.colorbar(label=f"F (relative, {args.units}/mol)")

        trans_mask = counts_trans >= args.trans_min_count
        if args.trans_overlay == "heatmap":
            with np.errstate(invalid="ignore"):
                trans_prob = counts_trans / np.sum(counts_trans) if np.sum(counts_trans) > 0 else counts_trans
            trans_prob[~trans_mask] = np.nan
            plt.imshow(trans_prob.T, origin="lower", aspect="auto", cmap="Reds", alpha=args.trans_alpha,
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        else:
            levels = np.linspace(np.nanmin(counts_trans[trans_mask]) if np.any(trans_mask) else 0,
                                 np.nanmax(counts_trans[trans_mask]) if np.any(trans_mask) else 1, 6)
            if np.any(trans_mask):
                plt.contour(counts_trans.T, levels=levels, colors="black",
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

        plt.xlabel(f"{label_cv1} (distance, A)")
        plt.ylabel(f"{label_cv2} (distance, A)")
        title = "Transition Density Over FES"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}{suffix}_transition_fes_overlay.png"), dpi=220)
        plt.close()

        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_counts_all.npy"), counts_all)
        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_counts_trans.npy"), counts_trans)
        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_xedges.npy"), xedges)
        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_yedges.npy"), yedges)

    # Shared bins across pooled and per-episode for comparability
    bins = compute_bins(cv1, cv2, bins=args.bins, bin_width=args.bin_width,
                        min_bins=args.min_bins, max_bins=args.max_bins, bin_strategy=args.bin_strategy)
    counts_all_pooled, xedges, yedges = np.histogram2d(cv1, cv2, bins=bins)
    if trans_idx.size > 0:
        counts_trans_pooled, _, _ = np.histogram2d(cv1[trans_idx], cv2[trans_idx], bins=[xedges, yedges])
    else:
        counts_trans_pooled = np.zeros_like(counts_all_pooled)

    if not args.no_pooled:
        _transition_scatter(trans_idx, "", title_suffix="pooled")
        _transition_overlay(counts_all_pooled, counts_trans_pooled, xedges, yedges, "", title_suffix="pooled")

    if args.per_episode and episode is not None:
        unique_eps = list(dict.fromkeys(episode.tolist()))
        if args.max_episodes is not None:
            unique_eps = unique_eps[: args.max_episodes]
        for ep in unique_eps:
            mask = episode == ep
            if not np.any(mask):
                continue
            ep_idx = np.where(mask)[0]
            trans_ep = trans_idx[np.isin(trans_idx, ep_idx)]
            suffix = f"_{format_episode_label(ep)}"
            _transition_scatter(trans_ep, suffix, title_suffix=f"{format_episode_label(ep)}")
            counts_all_ep, _, _ = np.histogram2d(cv1[mask], cv2[mask], bins=[xedges, yedges])
            if trans_ep.size > 0:
                counts_trans_ep, _, _ = np.histogram2d(cv1[trans_ep], cv2[trans_ep], bins=[xedges, yedges])
            else:
                counts_trans_ep = np.zeros_like(counts_all_ep)
            _transition_overlay(counts_all_ep, counts_trans_ep, xedges, yedges, suffix, title_suffix=f"{format_episode_label(ep)}")

    run_utils.cleanup_empty_dirs(run_dir)


if __name__ == "__main__":
    main()
