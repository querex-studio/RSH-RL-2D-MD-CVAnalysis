import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils


def compute_pca(data):
    mean = np.mean(data, axis=0)
    centered = data - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    scores = centered @ evecs
    explained = evals / np.sum(evals)
    return mean, evals, evecs, scores, explained


def save_plots(out_dir, scores, explained, dt_ps=1.0):
    os.makedirs(out_dir, exist_ok=True)

    # Scree plot
    plt.figure()
    plt.plot(np.arange(1, len(explained) + 1), explained, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA scree plot")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scree.png"))
    plt.close()

    # Cumulative variance
    plt.figure()
    plt.plot(np.arange(1, len(explained) + 1), np.cumsum(explained), marker="o")
    plt.xlabel("Component")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative variance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_cumulative.png"))
    plt.close()

    # PC1 vs PC2 scatter
    if scores.shape[1] >= 2:
        plt.figure()
        plt.scatter(scores[:, 0], scores[:, 1], s=4, alpha=0.6)
        plt.xlabel("PC1 (largest variance)")
        plt.ylabel("PC2 (2nd largest variance)")
        plt.title("PC1 vs PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pca_pc1_pc2.png"))
        plt.close()

    # PC time series
    t_ps = np.arange(scores.shape[0], dtype=np.float32) * float(dt_ps)
    plt.figure()
    plt.plot(t_ps, scores[:, 0], label="PC1")
    if scores.shape[1] >= 2:
        plt.plot(t_ps, scores[:, 1], label="PC2")
    plt.xlabel("Time (ps)")
    plt.ylabel("Score")
    plt.title("Principal component time series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_timeseries.png"))
    plt.close()


def compute_fes_2d(x, y, bins=60, kT=1.0):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)
    prob = hist / np.sum(hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -kT * np.log(prob)
    fes[~np.isfinite(fes)] = np.nan
    if np.any(np.isfinite(fes)):
        fes = fes - np.nanmin(fes)
    return fes, hist, xedges, yedges


def compute_fes_1d(x, bins=80, kT=1.0):
    hist, edges = np.histogram(x, bins=bins, density=False)
    prob = hist / np.sum(hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -kT * np.log(prob)
    fes[~np.isfinite(fes)] = np.nan
    if np.any(np.isfinite(fes)):
        fes = fes - np.nanmin(fes)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, fes, hist, edges


def plot_fes_2d(out_path, fes, xedges, yedges, title, start=None, target=None):
    plt.figure()
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(fes.T, origin="lower", extent=extent, cmap="coolwarm")
    plt.colorbar(label="F (relative)")
    if start is not None:
        plt.scatter(start[0], start[1], s=50, c="lime", edgecolor="black", label="start")
    if target is not None:
        plt.scatter(target[0], target[1], s=50, c="red", edgecolor="black", label="target")
    if start is not None or target is not None:
        plt.legend()
    plt.xlabel("PC1 (largest variance)")
    plt.ylabel("PC2 (2nd largest variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_fes_with_traj(out_path, fes, xedges, yedges, scores, title):
    plt.figure()
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(fes.T, origin="lower", extent=extent, cmap="coolwarm")
    plt.colorbar(label="F (relative)")
    plt.scatter(scores[:, 0], scores[:, 1], s=3, alpha=0.4, c="black")
    plt.xlabel("PC1 (largest variance)")
    plt.ylabel("PC2 (2nd largest variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter_time(out_path, scores, dt_ps=1.0):
    plt.figure()
    t_ps = np.arange(scores.shape[0], dtype=np.float32) * float(dt_ps)
    plt.scatter(scores[:, 0], scores[:, 1], c=t_ps, s=4, cmap="viridis")
    plt.colorbar(label="Time (ps)")
    plt.xlabel("PC1 (largest variance)")
    plt.ylabel("PC2 (2nd largest variance)")
    plt.title("PC1 vs PC2 (time colored)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_fes_1d(out_path, centers, fes, title, xlabel):
    plt.figure()
    plt.plot(centers, fes)
    plt.xlabel(xlabel)
    plt.ylabel("F (relative)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def interpret_variance(explained):
    if explained.size < 2:
        return "Only one component is available; 2D CV is not applicable."
    cumulative = float(explained[0] + explained[1])
    if cumulative >= 0.8:
        return "PC1+PC2 explain most variance; 2D CV is likely adequate."
    if cumulative >= 0.5:
        return "PC1+PC2 explain moderate variance; 2D CV may be usable with caution."
    return "PC1+PC2 explain limited variance; consider more PCs or nonlinear CVs."


def load_vector_from_npy(path, expected_dim):
    if path is None:
        return None
    arr = np.load(path)
    vec = np.array(arr).reshape(-1)
    if expected_dim is not None and vec.size != expected_dim:
        return None
    return vec


def main():
    parser = argparse.ArgumentParser(description="Run PCA on a coordinate array and generate plots.")
    parser.add_argument("--input", dest="input_path", default=None, help="Path to a .npy array (frames x dims).")
    parser.add_argument("--run", dest="run_dir", default=None, help="Existing run directory to store results.")
    parser.add_argument("--runs-root", default=None, help="Root folder that contains analysis_runs/")
    parser.add_argument("--bins", type=int, default=60, help="Number of bins for PCA FES.")
    parser.add_argument("--kT", type=float, default=1.0, help="kT for relative free energy in PCA FES.")
    parser.add_argument("--dt-ps", type=float, default=1.0, help="Time step per frame in ps (default: 1.0).")
    parser.add_argument("--start-npy", default=None, help="Optional .npy file for start vector.")
    parser.add_argument("--target-npy", default=None, help="Optional .npy file for target vector.")
    args = parser.parse_args()

    run_dir_arg = args.run_dir
    if run_dir_arg is not None and not os.path.isabs(run_dir_arg):
        run_dir_arg = os.path.join(ROOT_DIR, run_dir_arg)

    input_path = args.input_path
    if input_path is None:
        candidate = "sim_coordinate_6D.npy"
        if os.path.exists(candidate):
            input_path = candidate
        else:
            candidate = os.path.join(ROOT_DIR, "sim_coordinate_6D.npy")
            if os.path.exists(candidate):
                input_path = candidate

    if input_path is None:
        raise FileNotFoundError("No PCA input file found. Provide --input or place sim_coordinate_6D.npy.")

    data = np.load(input_path)
    mean, evals, evecs, scores, explained = compute_pca(data)

    if run_dir_arg:
        run_dir = run_dir_arg
        for subdir in ["data", os.path.join("figs", "analysis")]:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    else:
        time_tag = run_utils.default_time_tag()
        run_dir = run_utils.prepare_run_dir(time_tag, root=args.runs_root)
        run_utils.write_run_metadata(
            run_dir,
            {"script": "analysis/pca_analysis.py", "input": input_path},
        )

    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figs", "analysis"), exist_ok=True)

    np.save(os.path.join(run_dir, "data", "pca_mean.npy"), mean)
    np.save(os.path.join(run_dir, "data", "pca_evals.npy"), evals)
    np.save(os.path.join(run_dir, "data", "pca_evecs.npy"), evecs)
    np.save(os.path.join(run_dir, "data", "pca_scores.npy"), scores)
    np.save(os.path.join(run_dir, "data", "pca_explained.npy"), explained)

    fes2d = None
    hist2d = None
    xedges = None
    yedges = None
    pc1_centers = None
    pc1_fes = None
    pc2_centers = None
    pc2_fes = None

    if scores.shape[1] >= 2:
        # PCA -> FES in PC space
        fes2d, hist2d, xedges, yedges = compute_fes_2d(scores[:, 0], scores[:, 1], bins=args.bins, kT=args.kT)
        np.save(os.path.join(run_dir, "data", "pca_fes.npy"), fes2d)
        np.save(os.path.join(run_dir, "data", "pca_fes_hist.npy"), hist2d)
        np.save(os.path.join(run_dir, "data", "pca_fes_xedges.npy"), xedges)
        np.save(os.path.join(run_dir, "data", "pca_fes_yedges.npy"), yedges)

        pc1_centers, pc1_fes, pc1_hist, pc1_edges = compute_fes_1d(scores[:, 0], bins=args.bins, kT=args.kT)
        pc2_centers, pc2_fes, pc2_hist, pc2_edges = compute_fes_1d(scores[:, 1], bins=args.bins, kT=args.kT)
        np.save(os.path.join(run_dir, "data", "pca_pc1_fes.npy"), pc1_fes)
        np.save(os.path.join(run_dir, "data", "pca_pc2_fes.npy"), pc2_fes)

    # Start/target projection if dimensions match
    start_vec = load_vector_from_npy(args.start_npy, data.shape[1])
    target_vec = load_vector_from_npy(args.target_npy, data.shape[1])

    start_pc = None
    target_pc = None
    if scores.shape[1] >= 2:
        if start_vec is not None:
            start_scores = (start_vec - mean) @ evecs
            start_pc = (start_scores[0], start_scores[1])
            np.save(os.path.join(run_dir, "data", "pca_start_pc.npy"), np.array(start_pc))
        if target_vec is not None:
            target_scores = (target_vec - mean) @ evecs
            target_pc = (target_scores[0], target_scores[1])
            np.save(os.path.join(run_dir, "data", "pca_target_pc.npy"), np.array(target_pc))

    summary_path = os.path.join(run_dir, "data", "pca_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Explained variance ratio:\n")
        for i, val in enumerate(explained, start=1):
            f.write(f"PC{i}: {val:.6f}\n")
        f.write("\nKPIs:\n")
        f.write(f"PCA score dims: {scores.shape[1]}\n")
        f.write(f"PC1+PC2 cumulative: {float(explained[0] + explained[1]):.6f}\n")
        f.write(f"PC1 variance ratio: {float(explained[0]):.6f}\n")
        f.write(f"PC2 variance ratio: {float(explained[1]) if explained.size > 1 else 0.0:.6f}\n")
        f.write(f"FES bins: {args.bins}\n")
        if hist2d is not None:
            f.write(f"FES nonempty bins: {int(np.sum(hist2d > 0))}\n")
        if fes2d is not None and np.any(np.isfinite(fes2d)):
            f.write(f"FES range (relative): {float(np.nanmin(fes2d)):.6f} to {float(np.nanmax(fes2d)):.6f}\n")
        if start_pc is not None and target_pc is not None:
            dist = float(np.linalg.norm(np.array(start_pc) - np.array(target_pc)))
            f.write(f"Start-Target distance (PC space): {dist:.6f}\n")
        else:
            f.write("Start/Target projection: not available (dimension mismatch or missing input).\n")
        f.write("\nInterpretation:\n")
        f.write(interpret_variance(explained))

    report_path = os.path.join(run_dir, "data", "pca_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PCA Report\n\n")
        f.write(f"- Input shape: {data.shape}\n")
        f.write(f"- Bins: {args.bins}\n")
        f.write(f"- kT (relative): {args.kT}\n")
        f.write("\n## Explained variance\n\n")
        for i, val in enumerate(explained, start=1):
            f.write(f"- PC{i}: {val:.6f}\n")
        f.write("\n## KPIs\n\n")
        f.write(f"- PCA score dims: {scores.shape[1]}\n")
        f.write(f"- PC1+PC2 cumulative: {float(explained[0] + explained[1]):.6f}\n")
        if hist2d is not None:
            f.write(f"- FES nonempty bins: {int(np.sum(hist2d > 0))}\n")
        if fes2d is not None and np.any(np.isfinite(fes2d)):
            f.write(f"- FES range (relative): {float(np.nanmin(fes2d)):.6f} to {float(np.nanmax(fes2d)):.6f}\n")
        if start_pc is not None and target_pc is not None:
            dist = float(np.linalg.norm(np.array(start_pc) - np.array(target_pc)))
            f.write(f"- Start-target distance (PC space): {dist:.6f}\n")
        else:
            f.write("- Start/target projection: not available\n")
        f.write("\n## Interpretation\n\n")
        f.write(interpret_variance(explained))

    out_dir = os.path.join(run_dir, "figs", "analysis")
    save_plots(out_dir, scores, explained, dt_ps=args.dt_ps)
    if scores.shape[1] >= 2:
        plot_scatter_time(os.path.join(out_dir, "pca_pc1_pc2_time.png"), scores, dt_ps=args.dt_ps)
        plot_fes_2d(os.path.join(out_dir, "pca_fes.png"), fes2d, xedges, yedges, "PCA FES (PC1-PC2)")
        plot_fes_with_traj(
            os.path.join(out_dir, "pca_fes_with_traj.png"),
            fes2d,
            xedges,
            yedges,
            scores,
            "PCA FES with trajectory",
        )
        plot_fes_2d(
            os.path.join(out_dir, "pca_fes_start_target.png"),
            fes2d,
            xedges,
            yedges,
            "PCA FES with start/target",
            start=start_pc,
            target=target_pc,
        )
        plot_fes_1d(os.path.join(out_dir, "pca_pc1_fes.png"), pc1_centers, pc1_fes, "PC1 free energy", "PC1")
        plot_fes_1d(os.path.join(out_dir, "pca_pc2_fes.png"), pc2_centers, pc2_fes, "PC2 free energy", "PC2")

    run_utils.cleanup_empty_dirs(run_dir)


if __name__ == "__main__":
    main()
