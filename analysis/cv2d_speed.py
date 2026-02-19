import argparse
import json
import os
import sys
from typing import Tuple, Optional

import importlib
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils


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


def load_cv_from_run(run_dir) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
    return cv1, cv2, time


def load_cv_from_csv(path, cv1_col=None, cv2_col=None, time_col=None):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.dtype.names:
        names = list(arr.dtype.names)

        def _pick(col_name_candidates):
            for name in col_name_candidates:
                if name in names:
                    return arr[name]
            return None

        time = _pick(["time", "t", "step", "frame", "frames"])
        cv1 = _pick(["cv1", "cv1_A", "cv1_distance", "cv1_distance_A"])
        cv2 = _pick(["cv2", "cv2_A", "cv2_distance", "cv2_distance_A"])
        if cv1 is not None and cv2 is not None:
            return np.asarray(cv1), np.asarray(cv2), np.asarray(time) if time is not None else None

    raw = np.genfromtxt(path, delimiter=",")
    if raw.ndim == 1:
        raise ValueError("CSV appears to have a single row/column; cannot infer cv1/cv2.")
    if cv1_col is None or cv2_col is None:
        # default: time, cv1, cv2
        cv1_col = 1 if cv1_col is None else cv1_col
        cv2_col = 2 if cv2_col is None else cv2_col
    time = raw[:, time_col] if time_col is not None else (raw[:, 0] if raw.shape[1] >= 3 else None)
    return raw[:, cv1_col], raw[:, cv2_col], time


def finite_difference_speed(cv1, cv2, time=None, dt=1.0):
    cv1 = np.asarray(cv1).reshape(-1)
    cv2 = np.asarray(cv2).reshape(-1)
    if cv1.size != cv2.size:
        raise ValueError("cv1 and cv2 must have the same length.")
    if cv1.size < 2:
        raise ValueError("Need at least two samples to compute finite differences.")

    if time is None:
        dt_val = float(dt)
        dcv1 = np.diff(cv1) / dt_val
        dcv2 = np.diff(cv2) / dt_val
        speed = np.sqrt(dcv1 * dcv1 + dcv2 * dcv2)
        time_mid = (np.arange(cv1.size - 1) + 0.5) * dt_val
    else:
        time = np.asarray(time).reshape(-1)
        if time.size != cv1.size:
            raise ValueError("time and cv1/cv2 must have the same length.")
        dt_arr = np.diff(time)
        valid = dt_arr > 0
        dcv1 = np.full(cv1.size - 1, np.nan, dtype=float)
        dcv2 = np.full(cv2.size - 1, np.nan, dtype=float)
        dcv1[valid] = np.diff(cv1)[valid] / dt_arr[valid]
        dcv2[valid] = np.diff(cv2)[valid] / dt_arr[valid]
        speed = np.sqrt(dcv1 * dcv1 + dcv2 * dcv2)
        time_mid = (time[1:] + time[:-1]) / 2.0

    cv1_mid = (cv1[1:] + cv1[:-1]) / 2.0
    cv2_mid = (cv2[1:] + cv2[:-1]) / 2.0
    return time_mid, cv1_mid, cv2_mid, speed, dcv1, dcv2


def top_speed_spikes(time_mid, cv1_mid, cv2_mid, speed, top_n=20):
    finite = np.isfinite(speed)
    if not np.any(finite):
        return []
    idx = np.argsort(speed[finite])[::-1]
    finite_idx = np.nonzero(finite)[0]
    top_idx = finite_idx[idx[:top_n]]
    spikes = []
    for rank, i in enumerate(top_idx, start=1):
        spikes.append({
            "rank": rank,
            "segment_index": int(i),
            "time_mid": float(time_mid[i]),
            "cv1_mid": float(cv1_mid[i]),
            "cv2_mid": float(cv2_mid[i]),
            "speed": float(speed[i]),
        })
    return spikes




def main():
    parser = argparse.ArgumentParser(description="Time-derivative and speed analysis in CV space.")
    parser.add_argument("--run", default=None, help="Run directory with data/cv1.npy, data/cv2.npy.")
    parser.add_argument("--runs-root", default=None, help="Root folder for analysis_runs.")
    parser.add_argument("--cv1", dest="cv1_path", default=None, help="Optional path to cv1.npy.")
    parser.add_argument("--cv2", dest="cv2_path", default=None, help="Optional path to cv2.npy.")
    parser.add_argument("--time", dest="time_path", default=None, help="Optional path to time.npy.")
    parser.add_argument("--csv", default=None, help="CSV with time, cv1, cv2 columns.")
    parser.add_argument("--cv1-col", type=int, default=None, help="CSV column index for CV1 (0-based).")
    parser.add_argument("--cv2-col", type=int, default=None, help="CSV column index for CV2 (0-based).")
    parser.add_argument("--time-col", type=int, default=None, help="CSV column index for time (0-based).")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step if time is unavailable.")
    parser.add_argument("--time-units", default="step", help="Label for time units.")
    parser.add_argument("--top-n", type=int, default=20, help="Top N speed spikes to report.")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--point-alpha", type=float, default=0.8)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--overlay-top-n", action="store_true", help="Overlay top-N spike points on speed map.")
    parser.add_argument("--overlay-threshold", action="store_true", help="Overlay threshold-based spike regions.")
    parser.add_argument("--show-legend", action="store_true", help="Show legend explaining overlays.")
    parser.add_argument("--caption", action="store_true", help="Add a caption block explaining overlays.")
    parser.add_argument("--caption-fontsize", type=float, default=8.5)
    parser.add_argument("--spike-percentile", type=float, default=None, help="Percentile threshold for spike regions.")
    parser.add_argument("--spike-min-speed", type=float, default=None, help="Absolute speed threshold for spike regions.")
    parser.add_argument("--spike-bin-width", type=float, default=None, help="Bin width (A) for spike-region grid.")
    parser.add_argument("--spike-bins", type=int, default=None, help="Number of bins for spike-region grid.")
    parser.add_argument("--spike-min-count", type=int, default=3, help="Min count for spike-region grid.")
    parser.add_argument("--spike-fill-alpha", type=float, default=0.35)
    parser.add_argument("--top-marker-size", type=float, default=24.0)
    parser.add_argument("--out-prefix", default="cv2d_speed", help="Output filename prefix.")
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
            run_utils.write_run_metadata(run_dir, {"script": "analysis/cv2d_speed.py"})

    cfg_name = args.config_module or ("combined_2d" if os.path.exists(os.path.join(ROOT_DIR, "combined_2d.py")) else "config")
    try:
        cfg = importlib.import_module(cfg_name)
    except Exception:
        cfg = None

    if args.cv1_path and args.cv2_path:
        cv1 = np.load(args.cv1_path)
        cv2 = np.load(args.cv2_path)
        time = np.load(args.time_path) if args.time_path else None
    elif args.csv:
        cv1, cv2, time = load_cv_from_csv(args.csv, args.cv1_col, args.cv2_col, args.time_col)
    else:
        cv1, cv2, time = load_cv_from_run(run_dir)

    time_mid, cv1_mid, cv2_mid, speed, dcv1, dcv2 = finite_difference_speed(cv1, cv2, time=time, dt=args.dt)

    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    np.save(os.path.join(data_dir, f"{args.out_prefix}_time_mid.npy"), time_mid)
    np.save(os.path.join(data_dir, f"{args.out_prefix}_cv1_mid.npy"), cv1_mid)
    np.save(os.path.join(data_dir, f"{args.out_prefix}_cv2_mid.npy"), cv2_mid)
    np.save(os.path.join(data_dir, f"{args.out_prefix}_speed.npy"), speed)
    np.save(os.path.join(data_dir, f"{args.out_prefix}_dcv1_dt.npy"), dcv1)
    np.save(os.path.join(data_dir, f"{args.out_prefix}_dcv2_dt.npy"), dcv2)

    label_cv1 = args.cv1_label or (getattr(cfg, "CV1_LABEL", "CV1") if cfg else "CV1")
    label_cv2 = args.cv2_label or (getattr(cfg, "CV2_LABEL", "CV2") if cfg else "CV2")

    # Speed vs time
    plt.figure(figsize=(7, 3.8))
    plt.plot(time_mid, speed, linewidth=0.8, color="black")
    plt.xlabel(f"time ({args.time_units})")
    plt.ylabel("speed (CV units / time)")
    plt.title("CV Speed vs Time")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}_speed_vs_time.png"), dpi=220)
    plt.close()

    # CV-space scatter colored by speed
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(cv1_mid, cv2_mid, c=speed, s=args.point_size, alpha=args.point_alpha, cmap=args.cmap, linewidths=0)
    plt.colorbar(sc, label="speed (CV units / time)")
    plt.xlabel(f"{label_cv1} (distance, A)")
    plt.ylabel(f"{label_cv2} (distance, A)")
    plt.title("CV1 vs CV2 Colored by Speed")
    spike_threshold = None
    if args.spike_min_speed is not None:
        spike_threshold = float(args.spike_min_speed)
    else:
        perc = args.spike_percentile if args.spike_percentile is not None else 95.0
        finite = np.isfinite(speed)
        if np.any(finite):
            spike_threshold = float(np.percentile(speed[finite], perc))

    spike_mask = None
    if spike_threshold is not None:
        spike_mask = np.isfinite(speed) & (speed >= spike_threshold)

    spike_bins = None
    legend_handles = []
    if args.overlay_threshold and spike_mask is not None:
        bins = args.spike_bins
        if bins is None:
            if args.spike_bin_width is not None:
                bw = float(args.spike_bin_width)
                nx = int(np.ceil((cv1_mid.max() - cv1_mid.min()) / bw))
                ny = int(np.ceil((cv2_mid.max() - cv2_mid.min()) / bw))
                bins = (max(nx, 10), max(ny, 10))
            else:
                bins = (60, 60)
        spike_bins = bins
        counts_all, xedges, yedges = np.histogram2d(cv1_mid, cv2_mid, bins=bins)
        counts_spike, _, _ = np.histogram2d(cv1_mid[spike_mask], cv2_mid[spike_mask], bins=[xedges, yedges])
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = counts_spike / counts_all
        frac[counts_all < args.spike_min_count] = np.nan
        plt.imshow(frac.T, origin="lower", aspect="auto", cmap="Reds", alpha=args.spike_fill_alpha,
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        if args.show_legend:
            from matplotlib.patches import Patch
            legend_handles.append(Patch(facecolor="red", alpha=args.spike_fill_alpha, label="Spike region (fraction of points above threshold)"))

    if args.overlay_top_n:
        spikes = top_speed_spikes(time_mid, cv1_mid, cv2_mid, speed, top_n=args.top_n)
        if spikes:
            top_cv1 = [s["cv1_mid"] for s in spikes]
            top_cv2 = [s["cv2_mid"] for s in spikes]
            plt.scatter(top_cv1, top_cv2, s=args.top_marker_size, facecolors="none",
                        edgecolors="black", linewidths=1.2)
            if args.show_legend:
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None",
                                             markerfacecolor="none", markersize=6, label=f"Top-{args.top_n} speed spikes"))
    if args.show_legend and legend_handles:
        plt.legend(handles=legend_handles, loc="best", frameon=True)
    if args.caption:
        parts = ["Color: speed magnitude |dCV/dt|."]
        if args.overlay_threshold and spike_threshold is not None:
            if args.spike_min_speed is not None:
                parts.append(f"Red overlay: spike regions where speed >= {spike_threshold:g}.")
            else:
                perc = args.spike_percentile if args.spike_percentile is not None else 95.0
                parts.append(f"Red overlay: spike regions where speed >= {perc:g}th percentile (value {spike_threshold:g}).")
        if args.overlay_top_n:
            parts.append(f"Hollow circles: top-{args.top_n} speed spikes.")
        caption = " ".join(parts)
        fig = plt.gcf()
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=args.caption_fontsize, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}_cv_speed_map.png"), dpi=220)
    plt.close()

    spikes = top_speed_spikes(time_mid, cv1_mid, cv2_mid, speed, top_n=args.top_n)
    spikes_path = os.path.join(data_dir, f"{args.out_prefix}_speed_spikes.json")
    with open(spikes_path, "w", encoding="utf-8") as f:
        json.dump(spikes, f, indent=2)

    spikes_csv = os.path.join(data_dir, f"{args.out_prefix}_speed_spikes.csv")
    with open(spikes_csv, "w", encoding="utf-8") as f:
        f.write("rank,segment_index,time_mid,cv1_mid,cv2_mid,speed\n")
        for row in spikes:
            f.write(f"{row['rank']},{row['segment_index']},{row['time_mid']},"
                    f"{row['cv1_mid']},{row['cv2_mid']},{row['speed']}\n")

    meta = {
        "spike_percentile": args.spike_percentile,
        "spike_min_speed": args.spike_min_speed,
        "spike_threshold": float(spike_threshold) if spike_threshold is not None else None,
        "spike_min_count": args.spike_min_count,
        "spike_bins": list(spike_bins) if isinstance(spike_bins, (list, tuple)) else spike_bins,
        "spike_bin_width": args.spike_bin_width,
        "overlay_threshold": args.overlay_threshold,
        "overlay_top_n": args.overlay_top_n,
    }
    meta_path = os.path.join(data_dir, f"{args.out_prefix}_speed_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    run_utils.cleanup_empty_dirs(run_dir)


if __name__ == "__main__":
    main()
