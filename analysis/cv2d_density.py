import argparse
import os
import sys
from typing import Tuple, Optional

import importlib
import numpy as np
import matplotlib.pyplot as plt
import heapq

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


def load_cv_from_run(run_dir) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    cv1_path = os.path.join(run_dir, "data", "cv1.npy")
    cv2_path = os.path.join(run_dir, "data", "cv2.npy")
    if not (os.path.exists(cv1_path) and os.path.exists(cv2_path)):
        raise FileNotFoundError("cv1.npy / cv2.npy not found in run_dir/data.")
    cv1 = np.load(cv1_path)
    cv2 = np.load(cv2_path)
    episode = None
    for name in ("episode.npy", "episodes.npy", "ep.npy"):
        e_path = os.path.join(run_dir, "data", name)
        if os.path.exists(e_path):
            episode = np.load(e_path)
            break
    return cv1, cv2, episode


def load_cv_from_csv(path, cv1_col=None, cv2_col=None, episode_col=None):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.dtype.names:
        names = list(arr.dtype.names)
        def _pick(col_name_candidates):
            for name in col_name_candidates:
                if name in names:
                    return arr[name]
            return None
        episode = _pick(["episode", "ep", "episode_id", "traj"])
        cv1 = _pick(["cv1", "cv1_A", "cv1_distance", "cv1_distance_A"])
        cv2 = _pick(["cv2", "cv2_A", "cv2_distance", "cv2_distance_A"])
        if cv1 is not None and cv2 is not None:
            return np.asarray(cv1), np.asarray(cv2), np.asarray(episode) if episode is not None else None
    # fallback to numeric columns
    raw = np.genfromtxt(path, delimiter=",")
    if raw.ndim == 1:
        raise ValueError("CSV appears to have a single row/column; cannot infer cv1/cv2.")
    if cv1_col is None or cv2_col is None:
        # default: time, cv1, cv2
        cv1_col = 1
        cv2_col = 2
    episode = raw[:, episode_col] if episode_col is not None else None
    return raw[:, cv1_col], raw[:, cv2_col], episode


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


def compute_bins(
    x,
    y,
    bins=None,
    bin_width=None,
    min_bins=30,
    max_bins=200,
    bin_strategy="auto",
):
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

    # auto: prefer FD, fallback to Scott or Sturges if IQR collapses
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


def compute_prob_fes(cv1, cv2, bins, kT, epsilon, pseudocount, relative_fes=True):
    counts, xedges, yedges = np.histogram2d(cv1, cv2, bins=bins)
    if pseudocount > 0:
        counts_pc = counts + pseudocount
        prob = counts_pc / np.sum(counts_pc)
    else:
        prob = counts / np.sum(counts)
    fes = -kT * np.log(prob + epsilon)
    if relative_fes and np.isfinite(fes).any():
        fes = fes - np.nanmin(fes)
    return counts, prob, fes, xedges, yedges


def mask_by_count(arr, counts, min_count):
    masked = arr.copy()
    masked[counts < min_count] = np.nan
    return masked


def gap_mask_from_metric(metric, counts, min_count, gap_percentile=None, gap_max_value=None, tail="low"):
    if gap_max_value is not None:
        if tail == "high":
            return (counts >= min_count) & (metric >= gap_max_value)
        return (counts >= min_count) & (metric <= gap_max_value)
    if gap_percentile is None:
        return None
    valid = metric[counts >= min_count]
    if valid.size == 0:
        return None
    thresh = np.percentile(valid, gap_percentile)
    if tail == "high":
        return (counts >= min_count) & (metric >= thresh)
    return (counts >= min_count) & (metric <= thresh)


def parse_percentiles(value, default=None):
    if value is None:
        return default or []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [float(p) for p in parts]


def format_episode_label(ep):
    try:
        val = float(ep)
        if abs(val - int(val)) < 1e-6:
            return f"ep{int(val)}"
        return f"ep{str(val).replace('.', 'p')}"
    except Exception:
        return f"ep{str(ep)}"


def _split_path_by_dcut(x, y, dcut):
    if dcut is None:
        return [(x, y)]
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2:
        return [(x, y)]
    dx = np.diff(x)
    dy = np.diff(y)
    step = np.sqrt(dx * dx + dy * dy)
    breaks = np.where(step > dcut)[0]
    if breaks.size == 0:
        return [(x, y)]
    segments = []
    start = 0
    for idx in breaks:
        segments.append((x[start:idx + 1], y[start:idx + 1]))
        start = idx + 1
    if start < x.size:
        segments.append((x[start:], y[start:]))
    return [seg for seg in segments if seg[0].size > 1]


def _path_line_collection(x, y, cmap="viridis", linewidth=0.6, alpha=0.5, norm=None, tvals=None):
    from matplotlib.collections import LineCollection
    x = np.asarray(x)
    y = np.asarray(y)
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    if tvals is None:
        tvals = np.arange(len(x) - 1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=linewidth, alpha=alpha)
    lc.set_array(np.asarray(tvals))
    return lc


def _iqr_ellipse(ax, x, y, color="white", edgecolor="black", alpha=0.25):
    from matplotlib.patches import Ellipse
    q1x, q3x = np.percentile(x, [25, 75])
    q1y, q3y = np.percentile(y, [25, 75])
    width = q3x - q1x
    height = q3y - q1y
    cx = 0.5 * (q1x + q3x)
    cy = 0.5 * (q1y + q3y)
    if not np.isfinite(width) or not np.isfinite(height):
        return
    if width <= 0 and height <= 0:
        return
    ell = Ellipse((cx, cy), width=width, height=height, angle=0,
                  facecolor=color, edgecolor=edgecolor, alpha=alpha, linewidth=1.0)
    ax.add_patch(ell)


def _segments_from_indices(indices, x, y, dcut):
    if len(indices) < 2:
        return []
    if dcut is None:
        return [np.asarray(indices)]
    xi = x[indices]
    yi = y[indices]
    dx = np.diff(xi)
    dy = np.diff(yi)
    step = np.sqrt(dx * dx + dy * dy)
    breaks = np.where(step > dcut)[0]
    if breaks.size == 0:
        return [np.asarray(indices)]
    segments = []
    start = 0
    for b in breaks:
        seg = indices[start:b + 1]
        if len(seg) > 1:
            segments.append(np.asarray(seg))
        start = b + 1
    if start < len(indices):
        seg = indices[start:]
        if len(seg) > 1:
            segments.append(np.asarray(seg))
    return segments


def _parse_pair(value):
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected pair in format 'cv1,cv2'.")
    return float(parts[0]), float(parts[1])


def _coord_to_index(xedges, yedges, x, y):
    ix = int(np.searchsorted(xedges, x) - 1)
    iy = int(np.searchsorted(yedges, y) - 1)
    ix = int(np.clip(ix, 0, len(xedges) - 2))
    iy = int(np.clip(iy, 0, len(yedges) - 2))
    return ix, iy


def _dijkstra_path(cost, start, goal, allow_diagonal=True, step_penalty=0.0):
    nx, ny = cost.shape
    dist = np.full((nx, ny), np.inf, dtype=float)
    prev = np.full((nx, ny, 2), -1, dtype=int)
    dist[start] = 0.0
    heap = [(0.0, start[0], start[1])]
    if allow_diagonal:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d != dist[x, y]:
            continue
        if (x, y) == goal:
            break
        for dx, dy in neighbors:
            nx2 = x + dx
            ny2 = y + dy
            if nx2 < 0 or ny2 < 0 or nx2 >= nx or ny2 >= ny:
                continue
            step_len = np.sqrt(dx * dx + dy * dy)
            w = 0.5 * (cost[x, y] + cost[nx2, ny2]) + step_penalty * step_len
            nd = d + w
            if nd < dist[nx2, ny2]:
                dist[nx2, ny2] = nd
                prev[nx2, ny2] = (x, y)
                heapq.heappush(heap, (nd, nx2, ny2))
    # Reconstruct
    if np.all(prev[goal] == -1) and goal != start:
        return None
    path = [goal]
    cur = goal
    while cur != start:
        px, py = prev[cur]
        if px < 0:
            break
        cur = (int(px), int(py))
        path.append(cur)
    path.reverse()
    return path


def main():
    parser = argparse.ArgumentParser(description="2D probability density and FES for CV1/CV2.")
    parser.add_argument("--run", default=None, help="Run directory with data/cv1.npy, data/cv2.npy.")
    parser.add_argument("--runs-root", default=None, help="Root folder for analysis_runs.")
    parser.add_argument("--cv1", dest="cv1_path", default=None, help="Optional path to cv1.npy.")
    parser.add_argument("--cv2", dest="cv2_path", default=None, help="Optional path to cv2.npy.")
    parser.add_argument("--csv", default=None, help="CSV with time, cv1, cv2 columns.")
    parser.add_argument("--cv1-col", type=int, default=None, help="CSV column index for CV1 (0-based).")
    parser.add_argument("--cv2-col", type=int, default=None, help="CSV column index for CV2 (0-based).")
    parser.add_argument("--episode-col", type=int, default=None, help="CSV column index for episode (0-based).")
    parser.add_argument("--bins", type=int, default=None, help="Number of bins (applied to both axes).")
    parser.add_argument("--bin-width", type=float, default=None, help="Bin width (A) for both axes.")
    parser.add_argument("--min-bins", type=int, default=30)
    parser.add_argument("--max-bins", type=int, default=200)
    parser.add_argument(
        "--bin-strategy",
        choices=["auto", "fd", "scott", "sturges", "sqrt"],
        default="auto",
        help="Automatic binning strategy when --bins/--bin-width are not set.",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--units", choices=["kcal", "kJ"], default="kcal")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--pseudocount", type=float, default=0.0)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--gap-max-count", type=int, default=None)
    parser.add_argument("--gap-max-prob", type=float, default=None)
    parser.add_argument("--gap-min-fes", type=float, default=None)
    parser.add_argument("--gap-percentile", type=float, default=None)
    parser.add_argument("--gap-from", choices=["counts", "prob", "fes"], default="counts")
    parser.add_argument("--overlay-traj", action="store_true", help="Overlay trajectory points on plots.")
    parser.add_argument("--overlay-gap", action="store_true", help="Overlay gap regions on plots.")
    parser.add_argument("--fes-contours", default="50,75,90,95", help="Percentiles for FES contour lines.")
    parser.add_argument("--show-legend", action="store_true", help="Show legend explaining overlays/contours.")
    parser.add_argument("--caption", action="store_true", help="Add a caption block explaining overlays.")
    parser.add_argument("--caption-fontsize", type=float, default=8.5)
    parser.add_argument("--absolute-fes", action="store_true", help="Do not shift FES by its minimum.")
    parser.add_argument("--traj-size", type=float, default=2.0)
    parser.add_argument("--traj-alpha", type=float, default=0.15)
    parser.add_argument("--traj-line", action="store_true", help="Overlay trajectory line (time-ordered).")
    parser.add_argument("--traj-line-width", type=float, default=0.6)
    parser.add_argument("--traj-line-alpha", type=float, default=0.25)
    parser.add_argument("--path-plot", action="store_true", help="Create a separate FES plot with trajectory path.")
    parser.add_argument("--path-color", default="black", help="Path line color (use 'time' for gradient).")
    parser.add_argument("--path-cmap", default="viridis", help="Colormap for time-gradient path.")
    parser.add_argument("--path-break-dcut", type=float, default=None, help="Break path when delta exceeds d_cut.")
    parser.add_argument("--path-break-episode", action="store_true", help="Break path across episode boundaries.")
    parser.add_argument("--episode-centroid", action="store_true", help="Overlay per-episode centroids and drift arrows.")
    parser.add_argument("--centroid-method", choices=["mean", "median"], default="median")
    parser.add_argument("--centroid-ellipse", action="store_true", help="Draw IQR ellipse per episode.")
    parser.add_argument("--centroid-alpha", type=float, default=0.25)
    parser.add_argument("--centroid-color", default="white")
    parser.add_argument("--centroid-edge", default="black")
    parser.add_argument("--centroid-size", type=float, default=30.0)
    parser.add_argument("--centroid-arrow-color", default="yellow")
    parser.add_argument("--centroid-arrow-width", type=float, default=1.0)
    parser.add_argument("--centroid-arrow-scale", type=float, default=12.0)
    parser.add_argument("--mfep", action="store_true", help="Compute MFEP (Dijkstra) and overlay on FES.")
    parser.add_argument("--mfep-start", default=None, help="MFEP start CV pair: cv1,cv2")
    parser.add_argument("--mfep-end", default=None, help="MFEP end CV pair: cv1,cv2")
    parser.add_argument("--mfep-diagonal", action="store_true", help="Allow diagonal moves in MFEP grid.")
    parser.add_argument("--mfep-step-penalty", type=float, default=0.0, help="Add step-length penalty to MFEP cost.")
    parser.add_argument("--mfep-nan-penalty", type=float, default=1e6, help="Cost for NaN bins in MFEP.")
    parser.add_argument("--mfep-color", default="lime", help="MFEP line color.")
    parser.add_argument("--mfep-linewidth", type=float, default=2.0)
    parser.add_argument("--mfep-traj-alpha", type=float, default=0.25)
    parser.add_argument("--mfep-traj-size", type=float, default=4.0)
    parser.add_argument("--annotate-jumps", action="store_true", help="Annotate large jumps on the path plot.")
    parser.add_argument("--jump-percentile", type=float, default=99.0)
    parser.add_argument("--jump-min-delta", type=float, default=None)
    parser.add_argument("--jump-max", type=int, default=10)
    parser.add_argument("--jump-color", default="red")
    parser.add_argument("--jump-linewidth", type=float, default=1.0)
    parser.add_argument("--jump-labels", action="store_true")
    parser.add_argument("--jump-label-size", type=float, default=7.0)
    parser.add_argument("--annotate-traps", action="store_true", help="Annotate basin trapping on the path plot.")
    parser.add_argument("--trap-percentile", type=float, default=10.0)
    parser.add_argument("--trap-min-len", type=int, default=20)
    parser.add_argument("--trap-max", type=int, default=5)
    parser.add_argument("--trap-color", default="cyan")
    parser.add_argument("--trap-marker-size", type=float, default=36.0)
    parser.add_argument("--trap-labels", action="store_true")
    parser.add_argument("--trap-label-size", type=float, default=7.0)
    parser.add_argument("--per-episode", action="store_true", help="Also compute per-episode FES/plots.")
    parser.add_argument("--no-pooled", action="store_true", help="Skip pooled (overall) FES/plots.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes for per-episode outputs.")
    parser.add_argument("--out-prefix", default="cv2d", help="Output filename prefix.")
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
            run_utils.write_run_metadata(run_dir, {"script": "analysis/cv2d_density.py"})

    cfg_name = args.config_module or ("combined_2d" if os.path.exists(os.path.join(ROOT_DIR, "combined_2d.py")) else "config")
    try:
        cfg = importlib.import_module(cfg_name)
    except Exception:
        cfg = None

    if args.cv1_path and args.cv2_path:
        cv1 = np.load(args.cv1_path)
        cv2 = np.load(args.cv2_path)
        episode = None
    elif args.csv:
        cv1, cv2, episode = load_cv_from_csv(args.csv, args.cv1_col, args.cv2_col, args.episode_col)
    else:
        cv1, cv2, episode = load_cv_from_run(run_dir)

    cv1 = np.asarray(cv1).reshape(-1)
    cv2 = np.asarray(cv2).reshape(-1)
    if episode is not None:
        episode = np.asarray(episode).reshape(-1)
        if episode.size != cv1.size:
            raise ValueError("episode must have the same length as cv1/cv2.")

    bins = compute_bins(
        cv1,
        cv2,
        bins=args.bins,
        bin_width=args.bin_width,
        min_bins=args.min_bins,
        max_bins=args.max_bins,
        bin_strategy=args.bin_strategy,
    )
    kB = KB_KCAL if args.units == "kcal" else KB_KJ
    kT = kB * float(args.temperature)

    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    label_cv1 = args.cv1_label or (getattr(cfg, "CV1_LABEL", "CV1") if cfg else "CV1")
    label_cv2 = args.cv2_label or (getattr(cfg, "CV2_LABEL", "CV2") if cfg else "CV2")

    def _process_subset(sub_cv1, sub_cv2, suffix, sub_episode=None):
        counts, prob, fes, xedges, yedges = compute_prob_fes(
            sub_cv1,
            sub_cv2,
            bins=bins,
            kT=kT,
            epsilon=args.epsilon,
            pseudocount=args.pseudocount,
            relative_fes=not args.absolute_fes,
        )
        prob_m = mask_by_count(prob, counts, args.min_count)
        fes_m = mask_by_count(fes, counts, args.min_count)

        gap_percentile = args.gap_percentile
        if gap_percentile is None:
            gap_percentile = 95.0 if args.gap_from == "fes" else 5.0

        if args.gap_from == "prob":
            gap_metric = prob
            gap_max_value = args.gap_max_prob
            tail = "low"
        elif args.gap_from == "fes":
            gap_metric = fes
            gap_max_value = args.gap_min_fes
            tail = "high"
        else:
            gap_metric = counts
            gap_max_value = args.gap_max_count
            tail = "low"

        gap_mask = gap_mask_from_metric(
            gap_metric,
            counts,
            args.min_count,
            gap_percentile=gap_percentile,
            gap_max_value=gap_max_value,
            tail=tail,
        )

        contour_percentiles = parse_percentiles(args.fes_contours, default=[50, 75, 90, 95])

        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_counts.npy"), counts)
        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_prob.npy"), prob)
        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_fes.npy"), fes)
        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_xedges.npy"), xedges)
        np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_yedges.npy"), yedges)

        # Heatmap of P
        plt.figure(figsize=(6, 5))
        plt.imshow(prob_m.T, origin="lower", aspect="auto", cmap="magma",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(label="P(CV1, CV2)")
        plt.xlabel(f"{label_cv1} (distance, A)")
        plt.ylabel(f"{label_cv2} (distance, A)")
        plt.title("2D Probability Density P(CV1, CV2)")
        prob_handles = []
        if args.overlay_gap and gap_mask is not None:
            plt.contour(gap_mask.T, levels=[0.5], colors="cyan",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            if args.show_legend:
                from matplotlib.lines import Line2D
                prob_handles.append(Line2D([0], [0], color="cyan", lw=1.2, label="Gap mask"))
        if args.overlay_traj:
            plt.scatter(sub_cv1, sub_cv2, s=args.traj_size, alpha=args.traj_alpha, c="white", linewidths=0)
            if args.show_legend:
                from matplotlib.lines import Line2D
                prob_handles.append(Line2D([0], [0], marker="o", color="white", linestyle="None",
                                           markerfacecolor="white", markersize=4, label="Trajectory points"))
        if args.traj_line and sub_cv1.size > 1:
            plt.plot(sub_cv1, sub_cv2, color="white", linewidth=args.traj_line_width, alpha=args.traj_line_alpha)
        if args.show_legend and prob_handles:
            plt.legend(handles=prob_handles, loc="best", frameon=True)
        if args.caption:
            parts = ["Background: P(CV1, CV2)."]
            if args.overlay_gap and gap_mask is not None:
                parts.append("Cyan outline: gap mask.")
            if args.overlay_traj:
                parts.append("White points: trajectory samples.")
            if args.traj_line:
                parts.append("White line: time-ordered trajectory path.")
            caption = " ".join(parts)
            fig = plt.gcf()
            fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=args.caption_fontsize, wrap=True)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
        else:
            plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}{suffix}_prob.png"), dpi=220)
        plt.close()

        # Heatmap of F (clean)
        plt.figure(figsize=(6, 5))
        if np.isfinite(fes_m).any():
            vmin = np.nanmin(fes_m)
            vmax = np.nanpercentile(fes_m[np.isfinite(fes_m)], 95)
        else:
            vmin = None
            vmax = None
        plt.imshow(fes_m.T, origin="lower", aspect="auto", cmap="coolwarm",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   vmin=vmin, vmax=vmax)
        fes_label = f"F ({'relative' if not args.absolute_fes else 'absolute'}, {args.units}/mol)"
        plt.colorbar(label=fes_label)
        plt.xlabel(f"{label_cv1} (distance, A)")
        plt.ylabel(f"{label_cv2} (distance, A)")
        plt.title("2D Free Energy Surface F(CV1, CV2)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}{suffix}_fes.png"), dpi=220)
        plt.close()

        # FES overlays (contours/gap/trajectory) in a separate file
        if contour_percentiles or args.overlay_gap or args.overlay_traj or args.show_legend or args.caption:
            plt.figure(figsize=(6, 5))
            plt.imshow(fes_m.T, origin="lower", aspect="auto", cmap="coolwarm",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       vmin=vmin, vmax=vmax)
            plt.colorbar(label=fes_label)
            plt.xlabel(f"{label_cv1} (distance, A)")
            plt.ylabel(f"{label_cv2} (distance, A)")
            plt.title("2D Free Energy Surface F(CV1, CV2)")
            handles = []
            if contour_percentiles:
                valid = fes_m[np.isfinite(fes_m)]
                if valid.size > 0:
                    levels = [np.percentile(valid, p) for p in contour_percentiles]
                    levels = sorted(set(float(l) for l in levels if np.isfinite(l)))
                    if len(levels) >= 2:
                        plt.contour(fes_m.T, levels=levels, colors="white", linewidths=0.8,
                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                        if args.show_legend:
                            from matplotlib.lines import Line2D
                            handles.append(Line2D([0], [0], color="white", lw=1.2,
                                                  label=f"FES contours (percentiles: {','.join(map(str, contour_percentiles))})"))
            if args.overlay_gap and gap_mask is not None:
                plt.contour(gap_mask.T, levels=[0.5], colors="black", linewidths=1.0,
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                if args.show_legend:
                    from matplotlib.lines import Line2D
                    handles.append(Line2D([0], [0], color="black", lw=1.2, label="Gap mask"))
            if args.overlay_traj:
                plt.scatter(sub_cv1, sub_cv2, s=args.traj_size, alpha=args.traj_alpha, c="black", linewidths=0)
                if args.show_legend:
                    from matplotlib.lines import Line2D
                    handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None",
                                          markersize=4, label="Trajectory points"))
            if args.show_legend and handles:
                plt.legend(handles=handles, loc="best", frameon=True)
            if args.caption:
                parts = ["Background: FES (relative)."]
                if contour_percentiles:
                    parts.append(
                        f"White contours: FES percentile levels ({','.join(map(str, contour_percentiles))}); "
                        "higher percentiles indicate higher free-energy ridges."
                    )
                if args.overlay_gap and gap_mask is not None:
                    if gap_max_value is not None:
                        direction = ">=" if tail == "high" else "<="
                        parts.append(f"Black outline: gap mask from {args.gap_from} ({direction}{gap_max_value:g}).")
                    else:
                        direction = "high" if tail == "high" else "low"
                        parts.append(f"Black outline: gap mask from {args.gap_from} ({direction} {gap_percentile:g}th percentile).")
                if args.overlay_traj:
                    parts.append("Black points: trajectory samples.")
                caption = " ".join(parts)
                fig = plt.gcf()
                fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=args.caption_fontsize, wrap=True)
                plt.tight_layout(rect=[0, 0.05, 1, 1])
            else:
                plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}{suffix}_fes_overlay.png"), dpi=220)
            plt.close()

        # FES path plot with trajectory line and annotations (no legend/caption)
        do_path_plot = (
            args.path_plot
            or args.traj_line
            or args.annotate_jumps
            or args.annotate_traps
            or args.episode_centroid
        )
        if do_path_plot and sub_cv1.size > 1:
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            plt.imshow(fes_m.T, origin="lower", aspect="auto", cmap="coolwarm",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       vmin=vmin, vmax=vmax)
            plt.colorbar(label=fes_label)
            plt.xlabel(f"{label_cv1} (distance, A)")
            plt.ylabel(f"{label_cv2} (distance, A)")
            plt.title("2D Free Energy Surface + Trajectory Path")

            # Build index segments in time order, respecting episode boundaries and d_cut breaks
            n = sub_cv1.size
            all_indices = np.arange(n, dtype=int)
            segment_indices = []
            ep_arr = sub_episode
            if args.path_break_episode and ep_arr is not None:
                start = 0
                for i in range(1, n):
                    if ep_arr[i] != ep_arr[i - 1]:
                        segment_indices.extend(_segments_from_indices(all_indices[start:i], sub_cv1, sub_cv2, args.path_break_dcut))
                        start = i
                segment_indices.extend(_segments_from_indices(all_indices[start:n], sub_cv1, sub_cv2, args.path_break_dcut))
            else:
                segment_indices = _segments_from_indices(all_indices, sub_cv1, sub_cv2, args.path_break_dcut)

            # Time-gradient path or solid color
            if args.path_color == "time":
                tmin = float(np.min(all_indices))
                tmax = float(np.max(all_indices)) if n > 1 else float(np.min(all_indices) + 1)
                norm = plt.Normalize(vmin=tmin, vmax=tmax)
                for idxs in segment_indices:
                    sx = sub_cv1[idxs]
                    sy = sub_cv2[idxs]
                    tvals = idxs[:-1]
                    lc = _path_line_collection(
                        sx,
                        sy,
                        cmap=args.path_cmap,
                        linewidth=args.traj_line_width,
                        alpha=args.traj_line_alpha,
                        norm=norm,
                        tvals=tvals,
                    )
                    ax.add_collection(lc)
                sm = plt.cm.ScalarMappable(norm=norm, cmap=args.path_cmap)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label="time index")
            else:
                for idxs in segment_indices:
                    sx = sub_cv1[idxs]
                    sy = sub_cv2[idxs]
                    ax.plot(sx, sy, color=args.path_color, linewidth=args.traj_line_width, alpha=args.traj_line_alpha)

            # Episode centroids and drift arrows
            if args.episode_centroid and ep_arr is not None:
                unique_eps = list(dict.fromkeys(ep_arr.tolist()))
                centroids = []
                for ep in unique_eps:
                    mask = ep_arr == ep
                    if not np.any(mask):
                        continue
                    x = sub_cv1[mask]
                    y = sub_cv2[mask]
                    if args.centroid_method == "mean":
                        cx = float(np.nanmean(x))
                        cy = float(np.nanmean(y))
                    else:
                        cx = float(np.nanmedian(x))
                        cy = float(np.nanmedian(y))
                    centroids.append((cx, cy))
                    ax.scatter([cx], [cy], s=args.centroid_size, facecolors=args.centroid_color,
                               edgecolors=args.centroid_edge, linewidths=1.2, zorder=7)
                    if args.centroid_ellipse:
                        _iqr_ellipse(ax, x, y, color=args.centroid_color, edgecolor=args.centroid_edge,
                                     alpha=args.centroid_alpha)
                if len(centroids) >= 2:
                    for (x0, y0), (x1, y1) in zip(centroids[:-1], centroids[1:]):
                        from matplotlib.patches import FancyArrowPatch
                        arrow = FancyArrowPatch(
                            (x0, y0),
                            (x1, y1),
                            arrowstyle="-|>",
                            mutation_scale=args.centroid_arrow_scale,
                            color=args.centroid_arrow_color,
                            linewidth=args.centroid_arrow_width,
                            alpha=0.9,
                            zorder=8,
                        )
                        ax.add_patch(arrow)

            # Jump annotations
            if args.annotate_jumps:
                dx = np.diff(sub_cv1)
                dy = np.diff(sub_cv2)
                step = np.sqrt(dx * dx + dy * dy)
                finite = np.isfinite(step)
                if np.any(finite):
                    thr = np.percentile(step[finite], args.jump_percentile)
                    if args.jump_min_delta is not None:
                        thr = max(thr, float(args.jump_min_delta))
                    jump_idx = np.where(step >= thr)[0]
                    if jump_idx.size > 0:
                        jump_idx = jump_idx[np.argsort(step[jump_idx])[::-1]]
                        jump_idx = jump_idx[: args.jump_max]
                        for j, i in enumerate(jump_idx, start=1):
                            x0, y0 = sub_cv1[i], sub_cv2[i]
                            x1, y1 = sub_cv1[i + 1], sub_cv2[i + 1]
                            plt.plot([x0, x1], [y0, y1], color=args.jump_color, linewidth=args.jump_linewidth, alpha=0.9)
                            if args.jump_labels:
                                xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                                plt.text(xm, ym, f"J{j}", color=args.jump_color, fontsize=args.jump_label_size,
                                         ha="center", va="center")

            # Basin trapping annotations
            if args.annotate_traps:
                dx = np.diff(sub_cv1)
                dy = np.diff(sub_cv2)
                step = np.sqrt(dx * dx + dy * dy)
                finite = np.isfinite(step)
                if np.any(finite):
                    thr = np.percentile(step[finite], args.trap_percentile)
                    mask = step <= thr
                    runs = []
                    i = 0
                    while i < len(mask):
                        if not mask[i]:
                            i += 1
                            continue
                        start = i
                        while i < len(mask) and mask[i]:
                            i += 1
                        end = i - 1
                        run_len = end - start + 1
                        if run_len >= args.trap_min_len:
                            runs.append((start, end, run_len))
                    if runs:
                        runs.sort(key=lambda r: r[2], reverse=True)
                        runs = runs[: args.trap_max]
                        for k, (start, end, _) in enumerate(runs, start=1):
                            pts = slice(start, end + 2)
                            cx = float(np.nanmean(sub_cv1[pts]))
                            cy = float(np.nanmean(sub_cv2[pts]))
                            plt.scatter([cx], [cy], s=args.trap_marker_size, facecolors="none",
                                        edgecolors=args.trap_color, linewidths=1.2)
                            if args.trap_labels:
                                plt.text(cx, cy, f"B{k}", color=args.trap_color, fontsize=args.trap_label_size,
                                         ha="center", va="center")

            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}{suffix}_fes_path.png"), dpi=220)
            plt.close()

        # MFEP plot: FES + MFEP line + trajectory scatter (no line)
        if args.mfep and sub_cv1.size > 1:
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            plt.imshow(fes_m.T, origin="lower", aspect="auto", cmap="coolwarm",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       vmin=vmin, vmax=vmax)
            plt.colorbar(label=fes_label)
            plt.xlabel(f"{label_cv1} (distance, A)")
            plt.ylabel(f"{label_cv2} (distance, A)")
            plt.title("FES + MFEP (reaction corridor)")

            start_xy = _parse_pair(args.mfep_start) if args.mfep_start else (float(sub_cv1[0]), float(sub_cv2[0]))
            end_xy = _parse_pair(args.mfep_end) if args.mfep_end else (float(sub_cv1[-1]), float(sub_cv2[-1]))
            start_idx = _coord_to_index(xedges, yedges, start_xy[0], start_xy[1])
            end_idx = _coord_to_index(xedges, yedges, end_xy[0], end_xy[1])

            cost = np.array(fes_m, copy=True)
            cost[~np.isfinite(cost)] = float(args.mfep_nan_penalty)
            path = _dijkstra_path(cost, start_idx, end_idx, allow_diagonal=args.mfep_diagonal,
                                  step_penalty=args.mfep_step_penalty)
            if path:
                xcenters = 0.5 * (xedges[:-1] + xedges[1:])
                ycenters = 0.5 * (yedges[:-1] + yedges[1:])
                path_x = [xcenters[i] for i, _ in path]
                path_y = [ycenters[j] for _, j in path]
                ax.plot(path_x, path_y, color=args.mfep_color, linewidth=args.mfep_linewidth)
                # Save MFEP path
                np.save(os.path.join(data_dir, f"{args.out_prefix}{suffix}_mfep_path.npy"),
                        np.column_stack([path_x, path_y]))

            # Trajectory scatter only
            ax.scatter(sub_cv1, sub_cv2, s=args.mfep_traj_size, alpha=args.mfep_traj_alpha,
                       c="black", linewidths=0)

            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{args.out_prefix}{suffix}_fes_mfep.png"), dpi=220)
            plt.close()

        if gap_mask is not None:
            gap_valid = gap_mask & np.isfinite(fes_m)
            nongap_valid = (~gap_mask) & np.isfinite(fes_m) & (counts >= args.min_count)
            stats = {
                "gap_fraction": float(np.sum(gap_mask) / np.sum(counts >= args.min_count)) if np.sum(counts >= args.min_count) else None,
                "gap_mean_fes": float(np.nanmean(fes_m[gap_valid])) if np.any(gap_valid) else None,
                "non_gap_mean_fes": float(np.nanmean(fes_m[nongap_valid])) if np.any(nongap_valid) else None,
                "gap_median_fes": float(np.nanmedian(fes_m[gap_valid])) if np.any(gap_valid) else None,
                "non_gap_median_fes": float(np.nanmedian(fes_m[nongap_valid])) if np.any(nongap_valid) else None,
                "gap_from": args.gap_from,
                "gap_percentile": gap_percentile,
                "gap_threshold_value": float(gap_max_value) if gap_max_value is not None else None,
                "fes_contour_percentiles": contour_percentiles,
                "bins": [int(bins[0]), int(bins[1])],
            }
            out_stats = os.path.join(data_dir, f"{args.out_prefix}{suffix}_gap_stats.json")
            with open(out_stats, "w", encoding="utf-8") as f:
                import json
                json.dump(stats, f, indent=2)

    if not args.no_pooled:
        _process_subset(cv1, cv2, "", sub_episode=episode)

    if args.per_episode and episode is not None:
        unique_eps = list(dict.fromkeys(episode.tolist()))
        if args.max_episodes is not None:
            unique_eps = unique_eps[: args.max_episodes]
        for ep in unique_eps:
            mask = episode == ep
            if not np.any(mask):
                continue
            suffix = f"_{format_episode_label(ep)}"
            _process_subset(cv1[mask], cv2[mask], suffix, sub_episode=episode[mask])

    run_utils.cleanup_empty_dirs(run_dir)


if __name__ == "__main__":
    main()

