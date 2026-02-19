import argparse
import glob
import os
import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt

try:
    import MDAnalysis as mda
except Exception:
    mda = None

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils


def _resolve_runs_root(cfg, root=None):
    base = root or getattr(cfg, "RUNS_DIR", "analysis_runs")
    if not os.path.isabs(base):
        base = os.path.join(ROOT_DIR, base)
    return base


def _resolve_default_traj_glob(cfg):
    base = getattr(cfg, "RESULTS_TRAJ_DIR", os.path.join(ROOT_DIR, "results_PPO", "dcd_trajs"))
    if not os.path.isabs(base):
        base = os.path.join(ROOT_DIR, base)
    return os.path.join(base, "*.dcd")


def _snapshot_cfg(cfg):
    keys = [
        "ATOM1_INDEX", "ATOM2_INDEX", "ATOM3_INDEX", "ATOM4_INDEX",
        "CURRENT_DISTANCE", "FINAL_TARGET", "CURRENT_DISTANCE_2", "FINAL_TARGET_2",
        "TARGET_MIN", "TARGET_MAX", "TARGET2_MIN", "TARGET2_MAX",
        "stepsize", "dcdfreq_mfpt", "DCD_REPORT_INTERVAL",
        "RESULTS_DIR", "RESULTS_TRAJ_DIR",
    ]
    unit_mod = None
    if hasattr(cfg, "unit"):
        unit_mod = cfg.unit
    elif hasattr(cfg, "u"):
        unit_mod = cfg.u
    snap = {}
    for k in keys:
        if not hasattr(cfg, k):
            continue
        val = getattr(cfg, k)
        try:
            if hasattr(val, "value_in_unit") and unit_mod is not None:
                val = float(val.value_in_unit(unit_mod.picoseconds))
        except Exception:
            pass
        snap[k] = val
    return snap


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


def load_fes(run_dir):
    fes_path = os.path.join(run_dir, "data", "fes.npy")
    if os.path.exists(fes_path):
        return np.load(fes_path)
    return None


def plot_fes(fes, out_path, title="FES"):
    plt.figure()
    plt.imshow(fes, cmap="coolwarm", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _metrics_paths(run_dir):
    metrics_dir = os.path.join(run_dir, "metrics")
    if os.path.isdir(metrics_dir):
        return {
            "mfpt": os.path.join(metrics_dir, "total_steps_mfpt.csv"),
            "unbiased": os.path.join(metrics_dir, "total_steps_unbiased.csv"),
            "metad": os.path.join(metrics_dir, "total_steps_metaD.csv"),
        }
    # fallback: project root (legacy)
    return {
        "mfpt": os.path.join(ROOT_DIR, "total_steps_mfpt.csv"),
        "unbiased": os.path.join(ROOT_DIR, "total_steps_unbiased.csv"),
        "metad": os.path.join(ROOT_DIR, "total_steps_metaD.csv"),
    }


def plot_total_steps(run_dir, out_dir, cfg):
    paths = _metrics_paths(run_dir)
    data = {}
    unit_mod = None
    if hasattr(cfg, "unit"):
        unit_mod = cfg.unit
    elif hasattr(cfg, "u"):
        unit_mod = cfg.u
    if unit_mod is None or not hasattr(cfg, "stepsize"):
        return

    for key, path in paths.items():
        if os.path.exists(path):
            values = np.genfromtxt(path, delimiter=",")
            values = np.atleast_1d(values).reshape(-1)
            dt = float(cfg.stepsize.value_in_unit(unit_mod.picoseconds))
            if key == "unbiased" and hasattr(cfg, "stepsize_unbias"):
                dt = float(cfg.stepsize_unbias.value_in_unit(unit_mod.picoseconds))
            data[key] = values * dt

    if not data:
        return

    labels = []
    series = []
    for key in ["mfpt", "unbiased", "metad"]:
        if key in data:
            labels.append(key)
            series.append(data[key])

    plt.figure()
    plt.boxplot(series, labels=labels)
    plt.yscale("log")
    plt.ylabel("Time (ps)")
    plt.title("Time to reach destination")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "total_steps_boxplot_log.png"))
    plt.close()

    plt.figure()
    plt.violinplot(series)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.yscale("log")
    plt.ylabel("Time (ps)")
    plt.title("Time to reach destination")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "total_steps_violin_log.png"))
    plt.close()


def plot_reconstructed_fes(run_dir, out_dir, cfg):
    if not hasattr(cfg, "num_bins"):
        return
    paths = sorted(glob.glob(os.path.join(run_dir, "data", "*_reconstructed_fes_*.npy")))
    for path in paths:
        fes = np.load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        plot_fes(fes.reshape(cfg.num_bins, cfg.num_bins), os.path.join(out_dir, f"{name}.png"), title=name)


def plot_bias_surfaces(run_dir, out_dir, cfg):
    if not hasattr(cfg, "num_bins"):
        return
    param_paths = sorted(glob.glob(os.path.join(run_dir, "params", "*_gaussian_fes_param_*.txt")))
    if not param_paths:
        return
    x, y = np.meshgrid(np.linspace(0, 2 * np.pi, cfg.num_bins), np.linspace(0, 2 * np.pi, cfg.num_bins))
    for path in param_paths:
        params = np.loadtxt(path)
        from util import get_total_bias_2d
        total_bias = get_total_bias_2d(x, y, params)
        name = os.path.splitext(os.path.basename(path))[0]
        plot_fes(total_bias, os.path.join(out_dir, f"{name}_bias.png"), title=f"Bias {name}")


def plot_metaD_energy(run_dir, out_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "visited_states", "*_metaD_potential_energy.npy")))
    for path in paths:
        energy = np.load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        plt.figure()
        plt.plot(energy)
        plt.xlabel("Step")
        plt.ylabel("Potential energy (kJ/mol)")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close()


def plot_cv_trajectories(out_dir, traj_glob, top_path, cfg, max_plots=50, stride=1):
    if mda is None:
        return
    if not os.path.exists(top_path):
        return

    dcd_paths = sorted(glob.glob(traj_glob))
    if not dcd_paths:
        return
    dcd_paths = dcd_paths[:max_plots]

    atom1 = int(getattr(cfg, "ATOM1_INDEX", 0))
    atom2 = int(getattr(cfg, "ATOM2_INDEX", 0))
    atom3 = int(getattr(cfg, "ATOM3_INDEX", 0))
    atom4 = int(getattr(cfg, "ATOM4_INDEX", 0))

    unit_mod = None
    if hasattr(cfg, "unit"):
        unit_mod = cfg.unit
    elif hasattr(cfg, "u"):
        unit_mod = cfg.u
    if unit_mod is None or not hasattr(cfg, "stepsize"):
        return
    stepsize_ps = float(cfg.stepsize.value_in_unit(unit_mod.picoseconds))
    report_interval = int(getattr(cfg, "DCD_REPORT_INTERVAL", getattr(cfg, "dcdfreq_mfpt", 1)))
    dt_ps = stepsize_ps * report_interval * max(1, stride)

    for dcd_path in dcd_paths:
        u = mda.Universe(top_path, dcd_path)
        cv1 = []
        cv2 = []
        for i, ts in enumerate(u.trajectory):
            if stride > 1 and (i % stride) != 0:
                continue
            pos = u.atoms.positions.astype(np.float64)
            cv1.append(float(np.linalg.norm(pos[atom1] - pos[atom2])))
            cv2.append(float(np.linalg.norm(pos[atom3] - pos[atom4])))
        cv1 = np.asarray(cv1, dtype=np.float32)
        cv2 = np.asarray(cv2, dtype=np.float32)

        time_ps = np.arange(len(cv1), dtype=np.float32) * dt_ps
        base = os.path.splitext(os.path.basename(dcd_path))[0]

        # CV1
        plt.figure(figsize=(9, 4.5))
        plt.plot(time_ps, cv1, linewidth=1.6)
        plt.xlabel("Time (ps)")
        label_cv1 = getattr(cfg, "CV1_LABEL", "CV1 distance")
        plt.ylabel(f"{label_cv1} (Å)")
        plt.title(f"{base} CV1 trajectory")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_cv1.png"))
        plt.close()

        # CV2
        plt.figure(figsize=(9, 4.5))
        plt.plot(time_ps, cv2, linewidth=1.6)
        plt.xlabel("Time (ps)")
        label_cv2 = getattr(cfg, "CV2_LABEL", "CV2 distance")
        plt.ylabel(f"{label_cv2} (Å)")
        plt.title(f"{base} CV2 trajectory")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_cv2.png"))
        plt.close()

        # 2D path
        plt.figure(figsize=(6.5, 5.5))
        sc = plt.scatter(cv1, cv2, c=time_ps, s=8, cmap="viridis")
        cbar = plt.colorbar(sc)
        cbar.set_label("Time (ps)")
        label_cv1 = getattr(cfg, "CV1_LABEL", "CV1 distance")
        label_cv2 = getattr(cfg, "CV2_LABEL", "CV2 distance")
        plt.xlabel(f"{label_cv1} (Å)")
        plt.ylabel(f"{label_cv2} (Å)")
        plt.title(f"{base} CV1 vs CV2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_cv1_cv2.png"))
        plt.close()

        # CSV
        csv_path = os.path.join(out_dir, f"{base}_cv.csv")
        np.savetxt(
            csv_path,
            np.c_[time_ps, cv1, cv2],
            delimiter=",",
            header="time_ps,cv1_A,cv2_A",
            comments="",
        )


def main():
    parser = argparse.ArgumentParser(description="Post-process PPO trajectories and generate plots.")
    parser.add_argument("--config-module", default=None, help="Config module to use (default: combined_2d or config).")
    parser.add_argument("--run", dest="run_dir", default=None, help="Existing analysis run directory.")
    parser.add_argument("--runs-root", default=None, help="Root folder that contains analysis_runs/")
    parser.add_argument("--traj-glob", default=None, help="Glob for DCD trajectories.")
    parser.add_argument("--top", default=None, help="Topology file (PSF/PDB).")
    parser.add_argument("--max-traj-plots", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    cfg_name = args.config_module or ("combined_2d" if os.path.exists(os.path.join(ROOT_DIR, "combined_2d.py")) else "config")
    try:
        cfg = importlib.import_module(cfg_name)
    except Exception:
        cfg = importlib.import_module("config")

    runs_root = _resolve_runs_root(cfg, args.runs_root)
    run_dir = args.run_dir
    if run_dir is not None and not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT_DIR, run_dir)

    if run_dir is None:
        latest = find_latest_run(runs_root)
        if latest is None:
            time_tag = run_utils.default_time_tag()
            run_dir = run_utils.prepare_run_dir(time_tag, root=runs_root)
            run_utils.write_run_metadata(
                run_dir,
                {
                    "script": "analysis/post_process.py",
                    "config_module": cfg_name,
                    "config": _snapshot_cfg(cfg),
                },
            )
        else:
            run_dir = latest

    out_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    fes = load_fes(run_dir)
    if fes is not None:
        plot_fes(fes, os.path.join(out_dir, "fes.png"))

    plot_total_steps(run_dir, out_dir, cfg)
    plot_reconstructed_fes(run_dir, out_dir, cfg)
    plot_bias_surfaces(run_dir, out_dir, cfg)
    plot_metaD_energy(run_dir, out_dir)

    traj_glob = args.traj_glob or _resolve_default_traj_glob(cfg)
    top_path = args.top or getattr(cfg, "psf_file", None)
    top_path = os.path.join(ROOT_DIR, top_path) if top_path and not os.path.isabs(top_path) else top_path
    plot_cv_trajectories(out_dir, traj_glob, top_path, cfg, max_plots=args.max_traj_plots, stride=args.stride)

    run_utils.cleanup_empty_dirs(run_dir)


if __name__ == "__main__":
    main()
