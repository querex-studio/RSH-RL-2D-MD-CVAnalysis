"""
combined2.py (2D CV version)

Drop-in replacement for combined.py when you want *two* collective variables (CVs)
for protein RL training, while preserving the overall structure/flow:

- PPOAgent (actor/critic, same PPO update logic)
- ProteinEnvironmentRedesigned (OpenMM rollout, milestone/zone logic, per-step DCD)
- save_checkpoint / load via torch
- plot_distance_trajectory (now plots CV1 by default; also saves CV2 if present)

Key change vs 1D:
- Environment tracks two distances:
    CV1: distance(atom1, atom2)   (main progress CV)
    CV2: distance(atom3, atom4)   (auxiliary CV)
- Action still maps to (amplitude, width, offset) as before (same ACTION_SIZE).
- Bias is applied as the *sum of two 1D Gaussians* (one on CV1, one on CV2),
  sharing amplitude/width (split 50/50 to keep total scale comparable).

This keeps method signatures, agent logic, and main-loop usage unchanged.
"""

# ========================= Imports =========================

import os
import csv
import time
import uuid
import json
import random
import sys as _sys
import numpy as np
from datetime import datetime
from collections import deque, Counter
from tqdm import tqdm
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import openmm
from openmm import unit as u
import openmm.unit as omm_unit
from openmm.app import CharmmPsfFile, PDBFile, CharmmParameterSet
from openmm.app import Simulation, DCDReporter
from openmm.app import PME, HBonds, CutoffNonPeriodic
from openmm.app import NoCutoff

# ========================= Config (in-file) =========================

SEED = 42

# ---- OpenMM platform ----
class SliceableDeque(deque):
    """deque + supports list-style slicing (e.g. d[1:]) by materializing to list."""
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self)[idx]
        return super().__getitem__(idx)

def get_best_platform(verbose: bool = True):
    names = [
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    ]
    if verbose:
        print(f"Available OpenMM platforms: {names}")
    if "CUDA" in names:
        if verbose: print("Using CUDA platform (GPU)")
        return openmm.Platform.getPlatformByName("CUDA")
    if "OpenCL" in names:
        if verbose: print("Using OpenCL platform")
        return openmm.Platform.getPlatformByName("OpenCL")
    if verbose: print("Using CPU platform")
    return openmm.Platform.getPlatformByName("CPU")

# ---- Files ----
psf_file = "step3_input.psf"
pdb_file = "traj_0.restart.pdb"
toppar_file = "toppar.str"

# ---- 2D CV definitions (MUST set both pairs for true 2D) ----
# CV1 (primary progress CV)
ATOM1_INDEX = 7799
ATOM2_INDEX = 7840
CV1_LABEL = f"CV1 (atom {ATOM1_INDEX} - atom {ATOM2_INDEX} distance)"

# CV2 (auxiliary CV)  <-- YOU MUST SET THESE
ATOM3_INDEX = 487
ATOM4_INDEX = 3789
CV2_LABEL = f"CV2 (atom {ATOM3_INDEX} - atom {ATOM4_INDEX} distance)"

# Optional: override CV pairs via list (first two are used)
# e.g. ATOM_PAIRS = [(7799,7840), (123,456)]
ATOM_PAIRS = []

# ---- Targets (CV1) ----
CURRENT_DISTANCE = 3.3
FINAL_TARGET = 7.5
TARGET_CENTER = FINAL_TARGET
TARGET_ZONE_HALF_WIDTH = 0.35
TARGET_MIN = TARGET_CENTER - TARGET_ZONE_HALF_WIDTH
TARGET_MAX = TARGET_CENTER + TARGET_ZONE_HALF_WIDTH

# ---- Targets (CV2) ----
# If you do not know a good CV2 target yet:
#   - leave CURRENT_DISTANCE2 = None to auto-detect from the starting structure
#   - set TARGET2_ZONE_HALF_WIDTH to a reasonable corridor width (Å)
CURRENT_DISTANCE2 = 8.5
FINAL_TARGET2 = 4.0            # optional; if None, uses CURRENT_DISTANCE2
TARGET2_CENTER = FINAL_TARGET2           # optional; if None, uses FINAL_TARGET2 or CURRENT_DISTANCE2
TARGET2_ZONE_HALF_WIDTH = 0.35
TARGET2_MIN = TARGET2_CENTER - TARGET2_ZONE_HALF_WIDTH
TARGET2_MAX = TARGET2_CENTER + TARGET2_ZONE_HALF_WIDTH

# ---- Milestones ----
DISTANCE_INCREMENTS = [3.5, 3.8, 4.2, 5.0, 6.0, 7.0]
DISTANCE2_INCREMENTS = [8.4, 7.6, 6.8, 6.0, 5.2, 4.4]

# ---- Locks / confinement (CV1) ----
ENABLE_MILESTONE_LOCKS = False
LOCK_MARGIN = 0.15
BACKSTOP_K = 3.0e4
PERSIST_LOCKS_ACROSS_EPISODES = True
CARRY_STATE_ACROSS_EPISODES = True
FREEZE_EXPLORATION_AT_ZONE = False

# ---- Zone confinement (CV1) ----
ZONE_CONFINEMENT = True
ZONE_K = 8.0e4
ZONE_MARGIN_LOW = 0.05
ZONE_MARGIN_HIGH = 0.05

# ---- Zone confinement (CV2) ----
CV2_ZONE_CONFINEMENT = True
CV2_ZONE_K = 8.0e4
CV2_ZONE_MARGIN_LOW = 0.05
CV2_ZONE_MARGIN_HIGH = 0.05
CV2_AMP_FRACTION = 0.5
CV2_CENTER_RESTRAINT = False
CV2_CENTER_K = 5.0e3   # start here; tune 2e4–2e5

SEED_ZONE_CAP_IF_BEST_IN_ZONE = True

# ---- Observation/action ----
# 2D state adds 2 features (cv2 normalized + cv2 trend), so default is 10.
STATE_SIZE = 11

AMP_BINS = [0.0, 4.0, 8.0, 12.0, 16.0]
WIDTH_BINS = [0.3, 0.5, 0.7, 1.0]
OFFSET_BINS = [0.1, 0.2, 0.5, 1.0, 1.5]
ACTION_SIZE = len(AMP_BINS) * len(WIDTH_BINS) * len(OFFSET_BINS)

MIN_AMP, MAX_AMP = 0.0, 40.0
MIN_WIDTH, MAX_WIDTH = 0.1, 2.5
MAX_ESCALATION_FACTOR = 1.5
IN_ZONE_MAX_AMP = 1e9

# ---- PPO ----
N_STEPS = 8
BATCH_SIZE = 4
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.3
LR = 1e-4
PPO_TARGET_KL = 0.03

# ---- Episode & MD ----
MAX_ACTIONS_PER_EPISODE = 10
stepsize = 0.001 * u.picoseconds
fricCoef = 2.0 / u.picoseconds
T = 300 * u.kelvin

propagation_step = 4000
dcdfreq_mfpt = 40

# NaN recovery
MAX_INTEGRATOR_RETRIES = 2
MIN_STEPSIZE = 0.0005 * u.picoseconds

# ---- Rewards ----
PROGRESS_REWARD = 120.0
MILESTONE_REWARD = 200.0
BACKTRACK_PENALTY = -15.0
VELOCITY_BONUS = 10.0
STEP_PENALTY = -0.5

PHASE2_TOL = 0.08
CENTER_GAIN = 400.0
STABILITY_STEPS = 6
CONSISTENCY_BONUS = 50.0

# CV2 shaping (keeps auxiliary CV in corridor without dominating training)
CV2_DEVIATION_PENALTY = 0.0   # per Å outside CV2 zone

# ---- Curriculum / Eval ----
PROB_FRESH_START = 0.5
EVAL_EVERY = 5
N_EVAL_EPISODES = 3
SAVE_CHECKPOINT_EVERY = 5

ROOT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT_DIR, "results_PPO")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RUNS_DIR = os.path.join(RESULTS_DIR, "analysis_runs")
METRICS_CSV = f"{RESULTS_DIR}/training_metrics.csv"

EVAL_GREEDY = True

# ---- OpenMM system options ----
nonbondedCutoff = 1.0 * u.nanometer
backbone_constraint_strength = 100

# ---- DCD output ----
DCD_SAVE = True
RESULTS_TRAJ_DIR = os.path.join(RESULTS_DIR, "dcd_trajs")
DCD_REPORT_INTERVAL = dcdfreq_mfpt
RUN_NAME_PREFIX = "ep"
RUNS_TXT = os.path.join(RESULTS_TRAJ_DIR, "runs.txt")

# ---- Episode outputs ----
SAVE_EPISODE_PDB = True
EPISODE_PDB_EVERY = 1
EPISODE_PDB_DIR = os.path.join(RESULTS_DIR, "episode_pdbs")

SAVE_BIAS_PROFILE = True
BIAS_PROFILE_EVERY = 1
BIAS_PROFILE_DIR = os.path.join(RESULTS_DIR, "bias_profiles")
BIAS_PROFILE_BINS = 250
BIAS_PROFILE_PAD_SIGMA = 3.0

# --------------------- module self-alias (for parity with combined.py) ---------------------
config = _sys.modules[__name__]

# ========================= Small utilities =========================
def load_charmm_params(filename: str):
    """
    Load CHARMM parameters the same way as combined.py:
    - Read `toppar.str` line-by-line
    - Strip comments after '!'
    - Treat each non-empty line as a referenced parameter/topology file
    """
    par_files = []
    with open(filename, "r") as f:
        for line in f:
            line = line.split("!")[0].strip()
            if line:
                par_files.append(line)
    return CharmmParameterSet(*tuple(par_files))

def add_backbone_posres(system: openmm.System,
                        psf: CharmmPsfFile,
                        pdb: PDBFile,
                        strength: float,
                        skip_indices=None):
    """
    Same as combined.py: restrain backbone atoms (N, CA, C) to initial coordinates
    using a CustomExternalForce. Skip indices in skip_indices (e.g. CV atoms).
    """
    if skip_indices is None:
        skip_indices = set()
    else:
        skip_indices = set(skip_indices)

    force = openmm.CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", float(strength))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for i, pos in enumerate(pdb.positions):
        if i in skip_indices:
            continue
        # CharmmPsfFile exposes atom_list in OpenMM app
        if psf.atom_list[i].name in ("N", "CA", "C"):
            xyz = pos.value_in_unit(u.nanometer)
            force.addParticle(i, xyz)

    system.addForce(force)
    return force

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)
    return path

# ========================= Plotting / checkpoints =========================
# This section brings combined.py parity utilities into the 2D-CV module.
def _ensure(path: str) -> str:
    """Create directory (if needed) and return the path."""
    _ensure_dir(path)
    return path

# ======================== EPISODE EXPORT =====================
def export_episode_metadata(
    episode_num: int,
    bias_log,
    backstops_A,
    backstop_events,
) -> None:
    meta_dir = _ensure(f"{config.RESULTS_DIR}/episode_meta/")
    meta = {
        "episode": int(episode_num),
        "bias_log_columns": ["step", "cv_id", "kind", "amp_kcal", "center_A", "width_A"],
        "bias_log": [list(x) for x in bias_log],
        "backstops_A": list(map(float, backstops_A or [])),
        "backstop_events": [list(map(float, x)) for x in (backstop_events or [])],
        "start_A": float(config.CURRENT_DISTANCE),
        "target_center_A": float(config.TARGET_CENTER),
        "target_zone": [float(config.TARGET_MIN), float(config.TARGET_MAX)],
    }
    with open(os.path.join(meta_dir, f"episode_{episode_num:04d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved episode metadata JSON for episode {episode_num}.")

# ---------------------------------------------------------------------
# End-of-episode PDB writer (optional helper)
# ---------------------------------------------------------------------
def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _find_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return None

def _get_topology(env, simulation=None):
    if simulation is not None:
        topo = getattr(simulation, "topology", None)
        if topo is not None:
            return topo
    psf = getattr(env, "psf", None)
    if psf is not None and hasattr(psf, "topology"):
        return psf.topology
    topo = getattr(env, "_last_topology", None)
    if topo is not None:
        return topo
    raise RuntimeError("No Topology found (simulation.topology / env.psf.topology / env._last_topology).")

def _get_positions_from_simulation(simulation):
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)
    # Must be a Quantity with units
    if not hasattr(pos, "unit"):
        raise RuntimeError("Simulation positions lack units.")
    return pos

def _get_positions_from_context(context):
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)
    if not hasattr(pos, "unit"):
        raise RuntimeError("Context positions lack units.")
    return pos

def _coerce_positions_quantity(pos_array):
    # Wrap (N,3) array as Quantity in nm
    arr = np.asarray(pos_array)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr * u.nanometer
    raise RuntimeError("Cached positions are not a valid (N,3) array.")

def write_episode_pdb(env: "ProteinEnvironmentRedesigned", out_dir: str, episode_idx: int) -> str:
    _ensure_dir(out_dir)

    simulation = _find_attr(env, ("simulation", "sim", "_simulation", "_last_simulation"))
    context = None
    topo = None
    pos = None

    if simulation is not None:
        topo = _get_topology(env, simulation=simulation)
        try:
            pos = _get_positions_from_simulation(simulation)
        except Exception:
            pos = None

    if pos is None:
        context = _find_attr(env, ("context", "_context", "sim_context", "_last_context"))
        if context is not None and topo is None:
            topo = _get_topology(env, simulation=None)
        if context is not None and pos is None:
            try:
                pos = _get_positions_from_context(context)
            except Exception:
                pos = None

    if pos is None:
        cached = _find_attr(env, ("_last_positions", "current_positions", "positions_cache"))
        if cached is not None:
            pos = _coerce_positions_quantity(cached)

    if topo is None or pos is None:
        raise RuntimeError(
            "Could not assemble Topology+Positions. Checked Simulation/Context and cached positions. "
            "Ensure env caches _last_topology and _last_positions/current_positions."
        )

    tag = _now_tag()
    fname = os.path.join(out_dir, f"{tag}_episode_{episode_idx:04d}.pdb")
    with open(fname, "w") as fh:
        PDBFile.writeFile(topo, pos, fh, keepIds=True)

    print(f"[episode_pdb_writer] Saved end-of-episode PDB: {fname}")
    return fname

# ---------------------------------------------------------------------
# Metrics CSV helper
# ---------------------------------------------------------------------
def append_metrics_row(path: str, row_dict: Dict[str, Any]) -> None:
    _ensure(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------
def plot_distance_trajectory(
    episode_trajectories: List[List[float]],
    episode_num: int,
    distance_history: Optional[List[float]] = None,
    cv2_history: Optional[List[float]] = None,
    episode_trajectories_cv2: Optional[List[List[float]]] = None,
) -> None:
    """
    Plot full CV1 trajectory with the same time-axis logic as combined.py.
    If cv2_history is provided, also save CV2 plot/CSV.
    """
    plot_dir = _ensure(f"{config.RESULTS_DIR}/full_trajectories/")

    # Fallback to distance_history if no segments were captured
    if ((not episode_trajectories or len(episode_trajectories) == 0)
        and distance_history is not None and len(distance_history) > 1):
        episode_trajectories = [list(map(float, distance_history[1:]))]

    if not episode_trajectories:
        print(f"[Warning] No trajectory data recorded for episode {episode_num}. Nothing to plot.")
        return

    segs = [np.asarray(seg, dtype=np.float32) for seg in episode_trajectories if len(seg) > 0]
    full_traj = np.concatenate(segs) if len(segs) > 1 else segs[0]
    if full_traj.ndim == 0:
        full_traj = full_traj.reshape(1)

    dt_frame_ps = config.dcdfreq_mfpt * float(config.stepsize.value_in_unit(omm_unit.picoseconds))
    time_axis_ps = np.arange(len(full_traj), dtype=np.float32) * dt_frame_ps

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_axis_ps, full_traj, linewidth=1.8)
    plt.axhspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label="Target zone (CV1)")
    plt.axhline(config.TARGET_CENTER, linestyle="--", linewidth=1.0, label="Target center")
    plt.xlabel("Time (ps)")
    plt.ylabel(f"{config.CV1_LABEL} (Å)")
    plt.title(f"Episode {episode_num:04d} CV1 trajectory")
    plt.legend()
    out_png = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved trajectory plot: {out_png}")

    out_csv = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1.csv")
    np.savetxt(out_csv, np.c_[time_axis_ps, full_traj],
               delimiter=",", header="time_ps,cv1_distance_A", comments="")
    print(f"Saved trajectory CSV: {out_csv}")

    # ---- CV2 plotted exactly like CV1 (from per-chunk segments) ----
    if episode_trajectories_cv2:
        segs2 = [np.asarray(seg, dtype=np.float32) for seg in episode_trajectories_cv2 if len(seg) > 0]
        cv2_full = np.concatenate(segs2) if len(segs2) > 1 else segs2[0]
        if cv2_full.ndim == 0:
            cv2_full = cv2_full.reshape(1)

        # ---- 2D phase-space plot: CV1 vs CV2 colored by time ----
        n = int(min(len(full_traj), len(cv2_full)))
        if n > 1:
            x_cv1 = np.asarray(full_traj[:n], dtype=np.float32)
            y_cv2 = np.asarray(cv2_full[:n], dtype=np.float32)
            t_ps  = np.asarray(time_axis_ps[:n], dtype=np.float32)

            plt.figure(figsize=(6.5, 5.5))
            sc = plt.scatter(x_cv1, y_cv2, c=t_ps, s=10)
            cbar = plt.colorbar(sc)
            cbar.set_label("Time (ps)")

            plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.15)
            plt.axhspan(config.TARGET2_MIN, config.TARGET2_MAX, alpha=0.12)

            plt.xlabel(f"{config.CV1_LABEL} (Å)")
            plt.ylabel(f"{config.CV2_LABEL} (Å)")
            plt.title(f"Episode {episode_num:04d} CV1 vs CV2 (colored by time)")
            out2d_png = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1_cv2_2d.png")
            plt.tight_layout()
            plt.savefig(out2d_png, dpi=220)
            plt.close()
            print(f"Saved 2D CV plot: {out2d_png}")

            out2d_csv = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1_cv2_2d.csv")
            np.savetxt(
                out2d_csv,
                np.c_[t_ps, x_cv1, y_cv2],
                delimiter=",",
                header="time_ps,cv1_pi_mg_A,cv2_atom3_atom4_A",
                comments=""
            )
            print(f"Saved 2D CV CSV: {out2d_csv}")

        # ---- regular CV2 time-series plot/CSV (always runs when CV2 exists) ----
        dt_frame_ps = config.dcdfreq_mfpt * float(config.stepsize.value_in_unit(omm_unit.picoseconds))
        time_axis2_ps = np.arange(len(cv2_full), dtype=np.float32) * dt_frame_ps

        plt.figure(figsize=(9, 4.5))
        plt.plot(time_axis2_ps, cv2_full, linewidth=1.8)
        plt.axhspan(config.TARGET2_MIN, config.TARGET2_MAX, alpha=0.15, label="Target zone (CV2)")
        plt.axhline(config.TARGET2_CENTER, linestyle="--", linewidth=1.0, label="CV2 center")
        plt.xlabel("Time (ps)")
        plt.ylabel(f"{config.CV2_LABEL} (Å)")
        plt.title(f"Episode {episode_num:04d} CV2 trajectory")
        plt.legend()
        out2_png = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv2.png")
        plt.tight_layout()
        plt.savefig(out2_png, dpi=200)
        plt.close()
        print(f"Saved CV2 plot: {out2_png}")

        out2_csv = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv2.csv")
        np.savetxt(out2_csv, np.c_[time_axis2_ps, cv2_full],
                   delimiter=",", header="time_ps,cv2_distance_A", comments="")
        print(f"Saved CV2 CSV: {out2_csv}")

def _bias_profile_1d(biases, xgrid):
    total = np.zeros_like(xgrid, dtype=np.float64)
    for amp, center, width in biases:
        total += float(amp) * np.exp(-((xgrid - float(center)) ** 2) / (2.0 * float(width) ** 2))
    return total


def save_episode_bias_profiles(all_biases, episode_num: int):
    """
    Save total bias profile (sum of Gaussians) for CV1 and CV2.
    Units: distance in Å, bias in kcal/mol.
    """
    if not all_biases:
        return

    bias_cv1 = [(a, c, w) for (cv, a, c, w) in all_biases if cv == 1]
    bias_cv2 = [(a, c, w) for (cv, a, c, w) in all_biases if cv == 2]

    out_dir = _ensure(config.BIAS_PROFILE_DIR)

    def _plot_one(biases, label, default_lo, default_hi):
        if not biases:
            return
        centers = np.array([b[1] for b in biases], dtype=float)
        widths = np.array([max(1e-6, b[2]) for b in biases], dtype=float)
        pad = float(config.BIAS_PROFILE_PAD_SIGMA)
        lo = float(np.min(centers - pad * widths)) if centers.size else default_lo
        hi = float(np.max(centers + pad * widths)) if centers.size else default_hi
        lo = min(lo, default_lo)
        hi = max(hi, default_hi)
        x = np.linspace(lo, hi, int(config.BIAS_PROFILE_BINS))
        y = _bias_profile_1d(biases, x)

        png = os.path.join(out_dir, f"ep_{episode_num:04d}_{label}_bias_profile.png")
        csv_path = os.path.join(out_dir, f"ep_{episode_num:04d}_{label}_bias_profile.csv")

        plt.figure(figsize=(8, 4.5))
        plt.plot(x, y, linewidth=1.8)
        plt.xlabel(f"{label.upper()} distance (Å)")
        plt.ylabel("Bias (kcal/mol)")
        plt.title(f"Episode {episode_num:04d} {label.upper()} bias profile")
        plt.tight_layout()
        plt.savefig(png, dpi=200)
        plt.close()

        np.savetxt(
            csv_path,
            np.c_[x, y],
            delimiter=",",
            header=f"{label}_distance_A,bias_kcal_per_mol",
            comments="",
        )

    # Default ranges based on targets (fallback if bias centers are narrow)
    _plot_one(
        bias_cv1,
        "cv1",
        default_lo=max(0.5, float(config.CURRENT_DISTANCE) - 1.0),
        default_hi=float(config.FINAL_TARGET) + 1.0,
    )
    _plot_one(
        bias_cv2,
        "cv2",
        default_lo=max(0.5, float(config.FINAL_TARGET2) - 1.0),
        default_hi=float(config.CURRENT_DISTANCE2) + 1.0 if config.CURRENT_DISTANCE2 is not None else 10.0,
    )

def compute_coverage(dist_segments, bin_edges):
    if not dist_segments:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    arrs = [np.asarray(seg, dtype=float) for seg in dist_segments if len(seg) > 0]
    if not arrs:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    data = np.concatenate(arrs)
    hist, _ = np.histogram(data, bins=bin_edges)
    cov = hist.astype(float)
    if cov.sum() > 0:
        cov = cov / cov.sum()
    return cov

def plot_coverage_histogram(dist_segments, episode_num: int, bin_size=0.25):
    out_dir = _ensure(f"{config.RESULTS_DIR}/coverage/")
    lo = max(0.5, config.CURRENT_DISTANCE - 1.0)
    hi = config.FINAL_TARGET + 1.5
    bins = np.arange(lo, hi + bin_size, bin_size)
    cov = compute_coverage(dist_segments, bins)

    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.figure(figsize=(10, 4))
    plt.bar(centers, cov, width=bin_size * 0.9, align='center')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label='Target Zone')
    for m in config.DISTANCE_INCREMENTS:
        plt.axvline(x=m, linestyle=':', alpha=0.4)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Visit fraction')
    plt.title(f'Distance Coverage — Episode {episode_num:04d}')
    plt.legend(loc='upper right')
    out = os.path.join(out_dir, f"coverage_ep_{episode_num:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Saved coverage histogram: {out}")

def _bias_energy_components(bias_log, backstops_A, xA_grid):
    """
    Parity with combined.py:
      - returns per-bias energy curves and total energy curve (kJ/mol)
      - includes backstop walls if provided
    bias_log in combined_2d.py is (step, kind, amp_kcal, center_A, width_A).
    """
    r_nm = xA_grid / 10.0
    per_bias = []
    total = np.zeros_like(xA_grid, dtype=float)

    for entry in bias_log:
        if len(entry) == 6:
            _, cv_id, kind, amp_kcal, center_A, width_A = entry
        elif len(entry) == 5:
            # backward-compat
            _, kind, amp_kcal, center_A, width_A = entry
            cv_id = 1
        else:
            continue

        # Only decompose CV1 Gaussian terms
        if int(cv_id) != 1:
            continue
        if kind not in ("gaussian", "bias", "bias1"):
            continue

        A_kJ = float(amp_kcal) * 4.184
        mu_nm = float(center_A) / 10.0
        sig_nm = max(1e-6, float(width_A) / 10.0)
        e = A_kJ * np.exp(-((r_nm - mu_nm) ** 2) / (2.0 * sig_nm ** 2))
        per_bias.append(e)
        total += e

    if backstops_A:
        for m_eff_A in backstops_A:
            m_nm = float(m_eff_A) / 10.0
            mask = r_nm < m_nm
            e = np.zeros_like(xA_grid, dtype=float)
            e[mask] = float(config.BACKSTOP_K) * (m_nm - r_nm[mask]) ** 2
            total += e
            per_bias.append(e)

    return per_bias, total

def plot_bias_components_and_sum(bias_log, backstops_A, episode_num):
    if not bias_log and not backstops_A:
        return
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    lo = max(0.5, config.CURRENT_DISTANCE - 2.0)
    hi = config.FINAL_TARGET + 2.0
    xA = np.linspace(lo, hi, 1000)
    per_bias, total = _bias_energy_components(bias_log, backstops_A, xA)

    plt.figure(figsize=(12, 7))
    for i, e in enumerate(per_bias):
        label = (f"Bias {i+1}" if i < len(bias_log) else f"Backstop {i - len(bias_log) + 1}")
        plt.plot(xA, e, linewidth=1.5, alpha=0.9, label=label)
    plt.plot(xA, total, linewidth=2.5, alpha=1.0, label='Cumulative Bias')

    plt.axvline(x=config.CURRENT_DISTANCE, linestyle='--', linewidth=2, label='Start')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label='Target Zone')
    plt.axvline(x=config.TARGET_CENTER, linestyle='--', linewidth=3, label='Target Center')

    plt.xlabel('Position (Å)')
    plt.ylabel('Bias Energy (kJ/mol)')
    plt.title(f'Bias Potentials in Episode {episode_num}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    out = os.path.join(plot_dir, f"bias_components_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bias components plot: {out}")

def plot_bias_sum_only(bias_log, backstops_A, episode_num):
    if not bias_log and not backstops_A:
        return
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    lo = max(0.5, config.CURRENT_DISTANCE - 2.0)
    hi = config.FINAL_TARGET + 2.0
    xA = np.linspace(lo, hi, 1000)
    _, total = _bias_energy_components(bias_log, backstops_A, xA)

    plt.figure(figsize=(12, 7))
    plt.plot(xA, total, linewidth=2.5)
    plt.axvline(x=config.CURRENT_DISTANCE, linestyle='--', linewidth=2, label='Start')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label='Target Zone')
    plt.axvline(x=config.TARGET_CENTER, linestyle='--', linewidth=3, label='Target Center')

    idx_min = np.argmin(total)
    plt.scatter([xA[idx_min]], [total[idx_min]], s=60, label=f"Min: {xA[idx_min]:.2f} Å")

    plt.xlabel('Position (Å)')
    plt.ylabel('Bias Energy (kJ/mol)')
    plt.title(f'Cumulative Bias Landscape — Episode {episode_num:04d}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    out = os.path.join(plot_dir, f"bias_sum_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative bias plot: {out}")

def plot_bias_timeline(bias_log, backstop_events, episode_num):
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    if not bias_log:
        return
    steps = [int(e[0]) for e in bias_log]
    amps  = [float(e[3]) for e in bias_log] 
    plt.figure(figsize=(9, 4.0))
    plt.plot(steps, amps, linewidth=1.8)
    if backstop_events:
        for (s, _) in backstop_events:
            plt.axvline(int(s), linestyle="--", linewidth=1.0, alpha=0.5)
    plt.xlabel("RL step")
    plt.ylabel("Bias amplitude (kcal/mol)")
    plt.title(f"Episode {episode_num:04d} bias amplitude timeline")
    out = os.path.join(plot_dir, f"bias_timeline_ep_{episode_num:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_lock_snapshot(backstops_A, episode_num):
    if not backstops_A:
        return
    lock_dir = _ensure(f"{config.RESULTS_DIR}/locks/")
    plt.figure(figsize=(12, 3))
    for i, m in enumerate(backstops_A):
        plt.axvline(x=m, linestyle="--", label=f"Lock {i+1} @ {m:.2f} Å")
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label="Target Zone")
    plt.axvline(x=config.TARGET_CENTER, linestyle="--", linewidth=2, label="Target Center")
    plt.xlim(max(0.5, config.CURRENT_DISTANCE - 1.5), config.FINAL_TARGET + 1.5)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel("Position (Å)")
    plt.title(f"Milestone Locks Snapshot — Episode {episode_num}")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(lock_dir, f"locks_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved locks snapshot: {out}")

# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------
def save_checkpoint(agent: "PPOAgent", env: "ProteinEnvironmentRedesigned", ckpt_dir: str, episode: int) -> None:
    _ensure(ckpt_dir)
    path = os.path.join(ckpt_dir, f"ckpt_ep_{episode:04d}.pt")
    payload = {
        "episode": int(episode),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": agent.scheduler.state_dict(),
        "obs_norm": agent.obs_norm_state(),
        "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
        "env_meta": {
            "best_distance_ever": float(getattr(env, "best_distance_ever", 0.0)),
            "phase": int(getattr(env, "phase", 1)),
        },
    }
    try:
        torch.save(payload, path)
        print(f"Saved checkpoint: {path}")
    except Exception as e:
        print(f"[warn] failed to save checkpoint: {e}")

# ======================== Monitoring helpers =================
def _safe_import_mdanalysis():
    try:
        import MDAnalysis
        from MDAnalysis.analysis import distances
        return MDAnalysis, distances
    except Exception:
        print("[monitor] MDAnalysis not available; skipping trajectory analysis.")
        return None, None

def analyze_mg_coordination_for_dcd(psf_file, dcd_file, out_dir, run_name):
    MDAnalysis, distances = _safe_import_mdanalysis()
    if MDAnalysis is None:
        return None

    _ensure(out_dir)
    out_path = os.path.join(out_dir, f"{run_name}.txt")

    u = MDAnalysis.Universe(psf_file, dcd_file)
    mg_sel = u.select_atoms("resname MG")
    coordinating_atoms_updating = u.select_atoms("name O* and around 5 resname MG", updating=True)

    coord_counts = []
    site_counts = Counter()

    with open(out_path, "w") as output:
        for ts in u.trajectory:
            if len(coordinating_atoms_updating) == 0 or len(mg_sel) == 0:
                continue

            dist_arr = distances.distance_array(mg_sel, coordinating_atoms_updating)
            d_sorted = np.sort(dist_arr)[0]
            if len(d_sorted) == 0:
                continue
            if len(d_sorted) >= 7:
                cutoff = d_sorted[5] + (d_sorted[6] - d_sorted[5]) / 2.0
            elif len(d_sorted) >= 6:
                cutoff = d_sorted[5]
            else:
                cutoff = d_sorted[-1]

            temp_sele = u.select_atoms(f"name O* and around {cutoff} resname MG", updating=True)
            n = min(6, len(temp_sele))
            coord_counts.append(n)

            for ik in range(n):
                atom = temp_sele[ik]
                key = (atom.resid, atom.resname, atom.name)
                site_counts[key] += 1
                output.write(f"{atom.resid}_{atom.resname}_{atom.name}")
                output.write("," if ik < n - 1 else "\n")

    metrics = {}
    if coord_counts:
        arr = np.asarray(coord_counts, dtype=float)
        metrics["mg_coordination_mean"] = float(arr.mean())
        metrics["mg_coordination_std"] = float(arr.std())
        metrics["mg_unique_sites"] = int(len(site_counts))
    print(f"[monitor] Wrote Mg coordination file: {out_path}")
    return metrics


def analyze_pi_path_for_dcd(psf_file, dcd_file, out_dir, run_name):
    MDAnalysis, _ = _safe_import_mdanalysis()
    if MDAnalysis is None:
        return None

    _ensure(out_dir)
    out_path = os.path.join(out_dir, f"Pi-{run_name}.txt")

    u = MDAnalysis.Universe(psf_file, dcd_file)
    coord1 = u.select_atoms(
        "not segid HETC and around 2 (segid HETC and (name O2 or name O3))",
        updating=True
    )
    coord2 = u.select_atoms(
        "not segid HETC and around 2 (segid HETC and (name H1 or name H2))",
        updating=True
    )

    site_counts = Counter()
    frame_counts = []

    with open(out_path, "w") as output:
        for ts in u.trajectory:
            coordinating_atoms_updating = coord1 + coord2
            n_tot = len(coordinating_atoms_updating)
            frame_counts.append(n_tot)

            for ik in range(n_tot):
                atom = coordinating_atoms_updating[ik]
                key = (atom.resid, atom.resname, atom.name)
                site_counts[key] += 1
                output.write(f"{atom.resid}_{atom.resname}_{atom.name}")
                output.write("," if ik < n_tot - 1 else "\n")

    metrics = {}
    if frame_counts:
        arr = np.asarray(frame_counts, dtype=float)
        metrics["pi_contacts_mean"] = float(arr.mean())
        metrics["pi_contacts_std"] = float(arr.std())
        metrics["pi_unique_sites"] = int(len(site_counts))
    print(f"[monitor] Wrote Pi-path file: {out_path}")
    return metrics


def run_mdanalysis_monitoring(run_name, psf_file, dcd_file):
    if not os.path.isfile(dcd_file):
        print(f"[monitor] DCD file not found for run {run_name}: {dcd_file}")
        return None

    mg_metrics = analyze_mg_coordination_for_dcd(psf_file, dcd_file, config.MG_MONITOR_DIR, run_name)
    pi_metrics = analyze_pi_path_for_dcd(psf_file, dcd_file, config.PI_MONITOR_DIR, run_name)

    return {"mg": mg_metrics, "pi": pi_metrics}

# --------------------- public exports ---------------------
__all__ = [
    "config",
    "PPOAgent", "RunningNorm", "Actor", "Critic",
    "ProteinEnvironmentRedesigned",
    "write_episode_pdb",
    "_ensure", "append_metrics_row", "plot_distance_trajectory",
    "plot_coverage_histogram", "plot_bias_sum_only",
    "plot_bias_components_and_sum", "plot_bias_timeline",
    "plot_lock_snapshot", "export_episode_metadata",
    "save_checkpoint", "run_mdanalysis_monitoring",
]
class RunningNorm:
    def __init__(self):
        self.mean = None
        self.var = None
        self.count = 1e-8

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        if self.mean is None or self.var is None:
            return np.asarray(x, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32)) + 1e-8
        )

    def state_dict(self):
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "var": None if self.var is None else self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, d):
        if not d:
            return
        self.mean = None if d.get("mean") is None else np.asarray(d["mean"], dtype=np.float64)
        self.var = None if d.get("var") is None else np.asarray(d["var"], dtype=np.float64)
        self.count = float(d.get("count", 1e-8))

# ========================= PPO networks / agent =========================
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1=256, fc2=128, fc3=64):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, action_size)

    def forward_logits(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.clamp(self.fc4(x), -20, 20)

    def forward(self, state):
        return F.softmax(self.forward_logits(state), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_size, seed, fc1=256, fc2=128, fc3=64):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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

        self.exploration_noise = 0.1; self.min_exploration_noise = 0.01
        self.exploration_decay = 0.995

        self.obs_rms = RunningNorm()
        self.episode_count = 0

    def _mask_logits_if_needed(self, logits, state_batch):
        if logits.ndim == 1:
            logits = logits.unsqueeze(0); state_batch = state_batch.unsqueeze(0)
        mask_vals = []
        for s in state_batch.detach().cpu().numpy():
            cv1_in_zone = (s[5] >= 0.5)    # CV1 in-zone flag
            cv2_in_zone = (s[10] >= 0.5)   # CV2 in-zone flag (NEW)
            in_zone = (cv1_in_zone and cv2_in_zone) 
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
        if self.obs_rms.mean is None: self.obs_rms.update(s_np)
        norm_np = self.obs_rms.normalize(s_np).astype(np.float32)
        norm_state = torch.from_numpy(norm_np)

        with torch.no_grad():
            logits = self.actor.forward_logits(norm_state)
            logits = self._mask_logits_if_needed(logits, norm_state)
            probs = F.softmax(logits, dim=-1)
            value = self.critic(norm_state).item()

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / self.action_size

        if not training and config.EVAL_GREEDY:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(1, action.view(-1, 1)).squeeze(1) + 1e-8)
        else:
            if training and self.exploration_noise > self.min_exploration_noise and not config.FREEZE_EXPLORATION_AT_ZONE:
                noise = torch.randn_like(probs) * self.exploration_noise
                probs = F.softmax(torch.log(probs + 1e-8) + noise, dim=-1)
            dist = torch.distributions.Categorical(probs)
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

        self.obs_rms.update(states)
        norm_states = torch.from_numpy(self.obs_rms.normalize(states)).float()

        if self.memory[-1][5]:
            next_value = 0.0
        else:
            next_state = np.asarray(self.memory[-1][6], dtype=np.float32)
            if next_state.ndim == 1: next_state = next_state[None, :]
            self.obs_rms.update(next_state)
            ns = self.obs_rms.normalize(next_state).astype(np.float32)
            with torch.no_grad():
                next_value = self.critic(torch.from_numpy(ns)).item()

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = next_value if t == len(rewards)-1 else values[t+1]
            next_done = dones[t] if t == len(rewards)-1 else dones[t+1]
            delta = rewards[t] + config.GAMMA * next_v * (1 - next_done) - values[t]
            gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae; returns[t] = gae + values[t]

        return norm_states, actions, old_log_probs, advantages, returns

    def update(self):
        if len(self.memory) < config.N_STEPS: return {}
        data = self.compute_advantages()
        if data is None: return {}
        states, actions, old_log_probs, advantages, returns = data
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(self.memory)
        metrics = {'loss':0.0,'actor_loss':0.0,'critic_loss':0.0,'entropy':0.0,
                   'approx_kl':0.0,'clip_frac':0.0,'updates':0}

        for _ in range(config.N_EPOCHS):
            perm = torch.randperm(N)
            for i in range(0, N, config.BATCH_SIZE):
                idx = perm[i:i+config.BATCH_SIZE]
                if len(idx) < 2: continue
                b_states = states[idx]; b_actions = actions[idx]
                b_old = old_log_probs[idx]; b_adv = advantages[idx]; b_ret = returns[idx]

                logits = self.actor.forward_logits(b_states)
                logits = self._mask_logits_if_needed(logits, b_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_logp - b_old)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1 - config.CLIP_RANGE, 1 + config.CLIP_RANGE) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(b_states).squeeze()
                critic_loss = F.mse_loss(values, b_ret)

                loss = actor_loss + config.VF_COEF * critic_loss - config.ENT_COEF * entropy

                approx_kl = (b_old - new_logp).mean().item()
                clip_frac = (torch.abs(ratios - 1.0) > config.CLIP_RANGE).float().mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
                self.optimizer.step()

                metrics['loss'] += loss.item()
                metrics['actor_loss'] += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += approx_kl
                metrics['clip_frac'] += clip_frac
                metrics['updates'] += 1

                if approx_kl > config.PPO_TARGET_KL: break

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

# ========================= Environment helpers =========================
def _phase2_bowl_reward(d, center, half_width, gain):
    err = abs(d - center)
    if err >= half_width:
        return 0.0
    scale = 1.0 - (err / half_width) ** 2
    return gain * scale

def _nan_safe_propagate(
    sim: Simulation,
    nsteps: int,
    dcdfreq: int,
    prop_index: int,
    atom_pairs=None,            # [(a1,a2),(b1,b2)]
    out_cv1=None,               # list to append CV1 samples (Å)
    out_cv2=None,               # list to append CV2 samples (Å)
    out_times_ps=None,          # list to append times (ps)
    dt_min=u.picoseconds * 0.0005,
):
    """Run nsteps in dcdfreq chunks with NaN recovery and a tqdm progress bar.

    We step in chunks (simulation.step(dcdfreq)) so reporters fire consistently and we can show progress.
    """
    integ = sim.integrator
    orig_dt = integ.getStepSize()
    dt = orig_dt

    # ---- Parity with combined.py: local minimization + velocity re-init ----
    try:
        openmm.LocalEnergyMinimizer.minimize(
            sim.context, 10.0 * u.kilojoule_per_mole, 200
        )
    except Exception:
        pass
    try:
        sim.context.setVelocitiesToTemperature(T)
    except Exception:
        pass

    # number of chunks (at least 1)
    n_chunks = max(1, int(nsteps // max(1, int(dcdfreq))))
    chunk_steps = int(dcdfreq)
    t_ps = 0.0

    def _positions_finite():
        st = sim.context.getState(getPositions=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        return np.isfinite(np.asarray(pos)).all()

    retries_left = int(MAX_INTEGRATOR_RETRIES)

    for _ in tqdm(
        range(n_chunks),
        desc=f"MD Step {prop_index:>2d}",
        colour="red",
        ncols=80,
    ):
        try:
            sim.step(chunk_steps)
            if not _positions_finite():
                raise openmm.OpenMMException("NaN positions detected")
        except Exception:
            if retries_left <= 0:
                break
            retries_left -= 1
            dt = max(dt * 0.5, dt_min)
            try:
                integ.setStepSize(dt)
            except Exception:
                break
            # reinit velocities to recover
            try:
                sim.context.setVelocitiesToTemperature(T)
            except Exception:
                pass
            # retry once
            try:
                sim.step(chunk_steps)
                if not _positions_finite():
                    raise openmm.OpenMMException("NaN positions after recovery")
            except Exception:
                break

        t_ps += float(chunk_steps) * float(orig_dt.value_in_unit(u.picoseconds))
        if out_times_ps is not None:
            out_times_ps.append(float(t_ps))

        # --- record live CVs at the same cadence as CV1 plotting (every chunk) ---
        if atom_pairs is not None and (out_cv1 is not None or out_cv2 is not None):
            st_rec = sim.context.getState(getPositions=True)
            pos_nm = st_rec.getPositions(asNumpy=True).value_in_unit(u.nanometer)

            # CV1
            (a1, a2) = atom_pairs[0]
            d1_A = np.linalg.norm(pos_nm[a1] - pos_nm[a2]) * 10.0
            if out_cv1 is not None:
                out_cv1.append(float(d1_A))

            # CV2
            (b1, b2) = atom_pairs[1]
            d2_A = np.linalg.norm(pos_nm[b1] - pos_nm[b2]) * 10.0
            if out_cv2 is not None:
                out_cv2.append(float(d2_A))

    # restore original dt
    try:
        integ.setStepSize(orig_dt)
    except Exception:
        pass

# ========================= ProteinEnvironmentRedesigned (2D) =========================
class ProteinEnvironmentRedesigned:
    """
    Same environment contract as 1D, but with *two* CV distances.

    - CV1 drives milestones/phase transitions (default)
    - CV2 supplies additional shaping and can be biased as well
    """

    def __init__(self):
        # resolve CV pairs
        if ATOM_PAIRS and len(ATOM_PAIRS) >= 2:
            (a1, a2), (b1, b2) = ATOM_PAIRS[0], ATOM_PAIRS[1]
        else:
            a1, a2 = ATOM1_INDEX, ATOM2_INDEX
            b1, b2 = ATOM3_INDEX, ATOM4_INDEX

        if b1 is None or b2 is None:
            raise ValueError(
                "2D CV requires ATOM3_INDEX and ATOM4_INDEX (or ATOM_PAIRS with >=2 pairs)."
            )

        self.atom1_idx, self.atom2_idx = int(a1), int(a2)
        self.atom3_idx, self.atom4_idx = int(b1), int(b2)

        self.platform = None  # lazy init
        self.psf = None
        self.pdb = None
        self.params = None

        # pre-load topology/system XML once
        self._load_and_serialize_base_system()

        # episode / milestone bookkeeping
        self.milestones_hit = set()
        self.in_zone_count = 0

        # histories
        self.distance_history = SliceableDeque(maxlen=2000)
        self.distance2_history = SliceableDeque(maxlen=2000)

        self.bias_log = []
        self.all_biases_in_episode = []

        self.step_counter = 0

        # Phase & progress
        self.phase = 1
        self.in_zone_steps = 0
        self.no_improve_counter = 0

        # Locks
        self.locked_milestone_idx = -1
        self.backstops_A = []

        # MD state
        self.current_positions = None
        self.current_velocities = None

        # Zone walls cleared (CV1)
        self.zone_floor_A = None
        self.zone_ceiling_A = None

        # Zone walls cleared (CV2)
        self.zone2_floor_A = None
        self.zone2_ceiling_A = None

        # caches
        self.simulation = None
        self._last_topology = None
        self._last_positions = None

        # DCD bookkeeping
        self.current_episode_index = None
        self.current_dcd_index = 0
        self.current_dcd_paths = []
        self.current_run_name = None

        # targets for CV2 (resolved at reset from structure if needed)
        self._cv2_center = None
        self.cv2_center_on = False

        # per-episode live trajectories (sampled every dcdfreq_mfpt chunk)
        self.episode_trajectory_segments = []          # CV1 segments (already used by main.py)
        self.episode_trajectory_segments_cv2 = []      # CV2 segments (NEW)

        # initialize current distances from the start structure
        self.reset(seed_from_max_A=None, carry_state=False, episode_index=None)

    def _load_and_serialize_base_system(self):
        # do not create platform or Simulation here (import-safe)
        print("Setting up protein MD system...")
        psf = CharmmPsfFile(psf_file)
        pdb = PDBFile(pdb_file)
        params = load_charmm_params(toppar_file)

        # Match combined.py behavior: non-periodic system setup
        system = psf.createSystem(
            params,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=nonbondedCutoff,
            constraints=None,
        )

        # ---- Backbone positional restraints (parity with combined.py) ----
        add_backbone_posres(
            system,
            psf,
            pdb,
            backbone_constraint_strength,
            skip_indices={self.atom1_idx, self.atom2_idx, self.atom3_idx, self.atom4_idx},
        )

        self.psf = psf
        self.pdb = pdb
        self.params = params
        self.base_system_xml = openmm.XmlSerializer.serialize(system)
        print("Protein MD system setup complete.")

    # ---------- CV accessors ----------
    def _current_distances_A(self):
        """Compute (d1_A, d2_A) from the current context/positions."""
        if self.simulation is None:
            # construct a temporary context to measure from pdb positions
            plat = self.platform or get_best_platform(verbose=False)
            system = openmm.XmlSerializer.deserialize(self.base_system_xml)
            integ = openmm.LangevinIntegrator(T, fricCoef, stepsize)
            sim = Simulation(self.psf.topology, system, integ, plat)
            sim.context.setPositions(self._last_positions if self._last_positions is not None else self.pdb.positions)
            state = sim.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        else:
            state = self.simulation.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)

        p = pos
        d1 = np.linalg.norm(p[self.atom1_idx] - p[self.atom2_idx]) * 10.0
        d2 = np.linalg.norm(p[self.atom3_idx] - p[self.atom4_idx]) * 10.0
        return float(d1), float(d2)

    # ---------- State ----------
    def get_state(self):
        # normalize targets
        d1 = self.current_distance
        d2 = self.current_distance2

        self.distance_history.append(d1)
        self.distance2_history.append(d2)

        # trends
        if len(self.distance_history) >= 3:
            trend1 = (self.distance_history[-1] - self.distance_history[-3]) / 2.0
        else:
            trend1 = 0.0
        if len(self.distance2_history) >= 3:
            trend2 = (self.distance2_history[-1] - self.distance2_history[-3]) / 2.0
        else:
            trend2 = 0.0

        stability = 0.5
        if len(self.distance_history) >= 5:
            stability = 1.0 / (1.0 + np.std(list(self.distance_history)[-5:]))

        # --- anchored progress for BOTH CVs ---
        p1 = (d1 - CURRENT_DISTANCE) / max(1e-6, (FINAL_TARGET - CURRENT_DISTANCE))  # CV1 outward
        p1 = float(np.clip(p1, 0.0, 1.0))

        # CV2 inward (distance reduction): progress increases as d2 decreases
        den2 = max(1e-6, (CURRENT_DISTANCE2 - FINAL_TARGET2))
        p2 = (CURRENT_DISTANCE2 - d2) / den2
        p2 = float(np.clip(p2, 0.0, 1.0))

        # Require BOTH to progress: use the bottleneck
        overall = min(p1, p2)

        # CV2 zone indicator
        in_cv2_zone = 0.0
        if self._cv2_min is not None and self._cv2_max is not None:
            in_cv2_zone = float(self._cv2_min <= d2 <= self._cv2_max)

        state = np.array(
            [
                d1 / max(1e-6, FINAL_TARGET),                                # 0
                max(0.0, overall),                                           # 1
                abs(d1 - TARGET_CENTER) / max(1e-6, TARGET_ZONE_HALF_WIDTH), # 2
                np.clip(overall, 0.0, 1.0),                                  # 3
                trend1 / 0.1,                                                # 4
                float(TARGET_MIN <= d1 <= TARGET_MAX),                       # 5
                float(self.no_improve_counter > 0),                          # 6
                stability,                                                   # 7
                (d2 / max(1e-6, (self._cv2_center or max(1e-6, d2)))),       # 8
                trend2 / 0.1,                                                # 9
                in_cv2_zone,                                                 # 10
            ],
            dtype=np.float32,
        )
        return state

    # ---------- Forces ----------
    def _add_gaussian_force_1cv(self, system, amplitude_kcal, center_A, width_A, atom_i, atom_j):
        """
        Add ONE 1D Gaussian bias on the distance between (atom_i, atom_j).
        amplitude_kcal is interpreted in kcal/mol (converted to kJ/mol internally).
        center_A and width_A are in Å.
        """
        uid = str(uuid.uuid4())[:8]
        A_name = f"A_{uid}"
        mu_name = f"mu_{uid}"
        sig_name = f"sigma_{uid}"

        expr = f"{A_name}*exp(-((r-{mu_name})^2)/(2*{sig_name}^2))"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(A_name, float(amplitude_kcal) * 4.184)
        cf.addGlobalParameter(mu_name, float(center_A) / 10.0)
        cf.addGlobalParameter(sig_name, max(1e-6, float(width_A) / 10.0))
        cf.addBond(int(atom_i), int(atom_j))
        system.addForce(cf)
        return system

    def _add_gaussian_biases(self, system, atom_i, atom_j, biases):
        """
        Add MANY 1D Gaussian biases on the same distance as per-bond parameters.
        This avoids CUDA kernel parameter overflow from too many global parameters.
        """
        if not biases:
            return system

        expr = "A*exp(-((r-mu)^2)/(2*sigma^2))"
        cf = openmm.CustomBondForce(expr)
        cf.addPerBondParameter("A")
        cf.addPerBondParameter("mu")
        cf.addPerBondParameter("sigma")

        for amp_kcal, center_A, width_A in biases:
            cf.addBond(
                int(atom_i),
                int(atom_j),
                [
                    float(amp_kcal) * 4.184,
                    float(center_A) / 10.0,
                    max(1e-6, float(width_A) / 10.0),
                ],
            )

        system.addForce(cf)
        return system
    
    def _add_backstop_force(self, system, m_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_back_{uid}"
        mname = f"m_{uid}"
        expr = f"{kname}*({mname} - r)^2*step({mname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(BACKSTOP_K))
        cf.addGlobalParameter(mname, float(m_eff_A) / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_zone_upper_cap(self, system, u_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_cap_{uid}"
        uname = f"u_{uid}"
        expr = f"{kname}*(r - {uname})^2*step(r - {uname})"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(ZONE_K))
        cf.addGlobalParameter(uname, float(u_eff_A) / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_zone_lower_cap(self, system, l_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_floor_{uid}"
        lname = f"l_{uid}"
        expr = f"{kname}*({lname} - r)^2*step({lname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(ZONE_K))
        cf.addGlobalParameter(lname, float(l_eff_A) / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system
    
    def _add_zone_upper_cap_cv2(self, system, u_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k2_cap_{uid}"
        uname = f"u2_{uid}"
        expr = f"{kname}*(r - {uname})^2*step(r - {uname})"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(CV2_ZONE_K))
        cf.addGlobalParameter(uname, float(u_eff_A) / 10.0)
        cf.addBond(self.atom3_idx, self.atom4_idx)
        system.addForce(cf)
        return system

    def _add_zone_lower_cap_cv2(self, system, l_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k2_floor_{uid}"
        lname = f"l2_{uid}"
        expr = f"{kname}*({lname} - r)^2*step({lname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(CV2_ZONE_K))
        cf.addGlobalParameter(lname, float(l_eff_A) / 10.0)
        cf.addBond(self.atom3_idx, self.atom4_idx)
        system.addForce(cf)
        return system
    
    def _add_center_harmonic_cv2(self, system, center_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k2_center_{uid}"
        cname = f"c2_{uid}"
        expr = f"{kname}*(r - {cname})^2"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(CV2_CENTER_K))
        cf.addGlobalParameter(cname, float(center_A) / 10.0)
        cf.addBond(self.atom3_idx, self.atom4_idx)
        system.addForce(cf)
        return system

    def _system_with_all_forces(self):
        system = openmm.XmlSerializer.deserialize(self.base_system_xml)

        if ENABLE_MILESTONE_LOCKS:
            for m_eff_A in self.backstops_A:
                system = self._add_backstop_force(system, m_eff_A)

        if ZONE_CONFINEMENT:
            if self.zone_floor_A is not None:
                system = self._add_zone_lower_cap(system, self.zone_floor_A)
            if self.zone_ceiling_A is not None:
                system = self._add_zone_upper_cap(system, self.zone_ceiling_A)
        
        if CV2_ZONE_CONFINEMENT:
            if self.zone2_floor_A is not None:
                system = self._add_zone_lower_cap_cv2(system, self.zone2_floor_A)
            if self.zone2_ceiling_A is not None:
                system = self._add_zone_upper_cap_cv2(system, self.zone2_ceiling_A)

        if CV2_CENTER_RESTRAINT and getattr(self, "cv2_center_on", False):
            system = self._add_center_harmonic_cv2(system, self._cv2_center)

        bias_cv1 = []
        bias_cv2 = []
        for (cv_id, amp, posA, widthA) in self.all_biases_in_episode:
            if cv_id == 1:
                bias_cv1.append((amp, posA, widthA))
            elif cv_id == 2:
                bias_cv2.append((amp, posA, widthA))

        if bias_cv1:
            system = self._add_gaussian_biases(system, self.atom1_idx, self.atom2_idx, bias_cv1)
        if bias_cv2:
            system = self._add_gaussian_biases(system, self.atom3_idx, self.atom4_idx, bias_cv2)

        return system

    # ---------- action → Gaussian parameters ----------
    def smart_progressive_bias(self, action: int):
        # --- decode action bins (same as before) ---
        A_bins = AMP_BINS
        W_bins = WIDTH_BINS
        O_bins = OFFSET_BINS
        nW, nO = len(W_bins), len(O_bins)

        action = int(max(0, min(action, len(A_bins) * nW * nO - 1)))
        amp_idx = action // (nW * nO)
        rem = action % (nW * nO)
        width_idx = rem // nO
        off_idx = rem % nO

        base_amp = float(A_bins[amp_idx])
        base_width = float(W_bins[width_idx])
        base_offset = float(O_bins[off_idx])

        # --- anchored progress for BOTH CVs (match your get_state) ---
        p1 = (self.current_distance - CURRENT_DISTANCE) / max(1e-6, (FINAL_TARGET - CURRENT_DISTANCE))
        p1 = float(np.clip(p1, 0.0, 1.0))

        den2 = max(1e-6, (CURRENT_DISTANCE2 - FINAL_TARGET2))
        p2 = (CURRENT_DISTANCE2 - self.current_distance2) / den2
        p2 = float(np.clip(p2, 0.0, 1.0))

        progress = min(p1, p2)

        # shared scaling (same structure as your current CV1 logic)
        amp_base = base_amp * (3.0 - 2.0 * np.clip(progress, 0.0, 1.0))
        width = base_width * (1.5 - np.clip(progress, 0.0, 1.0))

        # escalation logic (keep same behavior)
        if self.no_improve_counter >= 2:
            escalation = min(1.0 + 0.7 * self.no_improve_counter, MAX_ESCALATION_FACTOR)
            amp_base *= escalation

        amp_base = float(np.clip(amp_base, MIN_AMP, MAX_AMP))
        width = float(np.clip(width, MIN_WIDTH, MAX_WIDTH))

        # ----- CV1 Gaussian: push OUTWARD (repel from center below current) -----
        center1 = self.current_distance - (base_offset + 0.3)
        center1 = float(np.clip(center1, 0.5, self.current_distance - 0.1))
        amp1 = amp_base

        # CV1 in-zone => minimal perturbation (your original behavior)
        if TARGET_MIN <= self.current_distance <= TARGET_MAX:
            amp1 = 0.0
            center1 = max(self.current_distance - 0.2, TARGET_MIN)
            width = 0.3

        # ----- CV2 Gaussian: push INWARD (repel from center above current) -----
        # opposite sign to CV1: center above current distance
        center2 = self.current_distance2 + (base_offset + 0.3)
        center2 = float(max(self.current_distance2 + 0.1, center2))
        center2 = float(np.clip(center2, 0.5, 50.0))
        amp2 = float(CV2_AMP_FRACTION) * amp_base  # let CV2 be scaled independently if desired

        # CV2 in-zone => minimal perturbation (CV2 equivalent of CV1 rule)
        if (self._cv2_min is not None) and (self._cv2_max is not None):
            if self._cv2_min <= self.current_distance2 <= self._cv2_max:
                amp2 = 0.0
                # mirror the CV1 "small center shift" concept but in the inward direction
                center2 = min(self.current_distance2 + 0.2, self._cv2_max)
                width = 0.3

        # Return TWO independent gaussians:
        # (cv_id, kind, amp_kcal, center_A, width_A)
        return (1, "gaussian", float(amp1), float(center1), float(width)), \
               (2, "gaussian", float(amp2), float(center2), float(width))

    # ---------- episode control ----------
    def _seed_persistent_locks(self, seed_from_max_A):
        # derive backstops from milestones below the seed distance
        self.backstops_A = []
        for m in DISTANCE_INCREMENTS:
            if m <= seed_from_max_A - LOCK_MARGIN:
                self.backstops_A.append(m - LOCK_MARGIN)

    def reset(self, seed_from_max_A=None, carry_state=False, episode_index=None):
        # set / clear milestone bookkeeping
        self.milestones_hit = set()
        self.milestones_hit_cv2 = set()
        self.in_zone_count = 0
        self.step_counter = 0
        self._cv2_bias_center = None
        self.cv2_center_on = False

        self.phase = 1
        self.in_zone_steps = 0
        self.no_improve_counter = 0

        # Locks
        self.locked_milestone_idx = -1
        self.backstops_A = []

        # MD state
        if not carry_state:
            self.current_positions = None
            self._last_positions = None
        else:
            # keep positions across episodes if you explicitly requested carry_state
            self._last_positions = self.current_positions
            
        self.current_velocities = None   # never carry velocities
        self._last_topology = None

        # Zone walls cleared (CV1)
        self.zone_floor_A = None
        self.zone_ceiling_A = None

        # Clear CV2 zone walls too
        self.zone2_floor_A = None
        self.zone2_ceiling_A = None

        # caches cleared
        self.simulation = None

        # Episode / DCD bookkeeping
        self.current_episode_index = episode_index
        self.current_dcd_index = 0
        self.current_dcd_paths = []
        if episode_index is None:
            self.current_run_name = None

        # clear per-episode trajectory segments
        self.episode_trajectory_segments = []
        self.episode_trajectory_segments_cv2 = []

        # seed cross-episode locks
        if seed_from_max_A is not None:
            self._seed_persistent_locks(seed_from_max_A)
            if SEED_ZONE_CAP_IF_BEST_IN_ZONE:
                if TARGET_MIN <= seed_from_max_A <= TARGET_MAX:
                    self.zone_floor_A = (TARGET_MIN + ZONE_MARGIN_LOW)
                    self.zone_ceiling_A = (TARGET_MAX - ZONE_MARGIN_HIGH)

        # runs.txt bookkeeping
        if DCD_SAVE and episode_index is not None:
            dcd_dir = RESULTS_TRAJ_DIR
            _ensure_dir(dcd_dir)
            run_name = f"{RUN_NAME_PREFIX}{episode_index:04d}"
            self.current_run_name = run_name
            runs_txt = RUNS_TXT
            existing = set()
            if os.path.exists(runs_txt):
                with open(runs_txt, "r") as fh:
                    existing = {ln.strip() for ln in fh if ln.strip()}
            if run_name not in existing:
                with open(runs_txt, "a") as fh:
                    fh.write(run_name + "\n")

        # resolve platform lazily
        if self.platform is None:
            self.platform = get_best_platform(verbose=True)

        # initialize distances
        d1, d2 = self._current_distances_A()
        self.current_distance = float(d1)
        self.current_distance2 = float(d2)

        # resolve CV2 targets now
        c2 = CURRENT_DISTANCE2 if CURRENT_DISTANCE2 is not None else self.current_distance2
        f2 = FINAL_TARGET2 if FINAL_TARGET2 is not None else c2
        center2 = TARGET2_CENTER if TARGET2_CENTER is not None else f2
        self._cv2_center = float(center2)
        self._cv2_min = float(center2 - TARGET2_ZONE_HALF_WIDTH) if TARGET2_ZONE_HALF_WIDTH is not None else None
        self._cv2_max = float(center2 + TARGET2_ZONE_HALF_WIDTH) if TARGET2_ZONE_HALF_WIDTH is not None else None

        # --- mirror CV1 "seed-best-in-zone" cap logic for CV2 ---
        if SEED_ZONE_CAP_IF_BEST_IN_ZONE:
            if (self._cv2_min is not None) and (self._cv2_max is not None):
                if self._cv2_min <= self.current_distance2 <= self._cv2_max:
                    self.zone2_floor_A   = self._cv2_min + CV2_ZONE_MARGIN_LOW
                    self.zone2_ceiling_A = self._cv2_max - CV2_ZONE_MARGIN_HIGH       

        return self.get_state()

    # ---------- step ----------
    def step(self, action_index):
        action_index = int(action_index)

        biases = self.smart_progressive_bias(action_index)

        self.step_counter += 1
        for b in biases:
            # Accept both legacy 4-tuples and new 5-tuples
            if len(b) == 4:
                cv_id, amp_kcal, center_A, width_A = b
                kind = "gaussian"
            elif len(b) == 5:
                cv_id, kind, amp_kcal, center_A, width_A = b
            else:
                raise ValueError(f"Unexpected bias tuple length={len(b)}: {b}")

            amp_kcal = float(min(float(amp_kcal), 12.0))
            width_A  = float(max(float(width_A), 0.3))

            self.all_biases_in_episode.append(
                (int(cv_id), float(amp_kcal), float(center_A), float(width_A))
            )

            # bias_log keeps (step, cv_id, kind, amp, center, width)
            self.bias_log.append(
                (self.step_counter, int(cv_id), str(kind), float(amp_kcal), float(center_A), float(width_A))
            )

        # build system with all forces and run MD
        system = self._system_with_all_forces()
        gamma = fricCoef
        if getattr(self, "cv2_center_on", False) or (self.phase == 2):
            gamma = 10.0 / u.picoseconds  # stronger damping only in the locked regime
        integrator = openmm.LangevinIntegrator(T, gamma, stepsize)

        sim = Simulation(self.psf.topology, system, integrator, self.platform)
        self.simulation = sim

        # Positions: carry forward like combined.py
        if self._last_positions is not None:
            sim.context.setPositions(self._last_positions)
        else:
            sim.context.setPositions(self.pdb.positions)

        #local minimization before launching dynamics
        try:
            openmm.LocalEnergyMinimizer.minimize(sim.context, 10.0 * u.kilojoule_per_mole, 200)
        except Exception:
            pass
        
        # Velocities: ALWAYS reinitialize (do NOT carry)
        sim.context.setVelocitiesToTemperature(T, SEED)

        # DCD reporter per RL action
        if DCD_SAVE and self.current_run_name is not None:
            dcd_dir = RESULTS_TRAJ_DIR
            _ensure_dir(dcd_dir)
            self.current_dcd_index += 1
            dcd_name = f"{self.current_run_name}_s{self.current_dcd_index:03d}.dcd"
            dcd_path = os.path.join(dcd_dir, dcd_name)
            sim.reporters.append(DCDReporter(dcd_path, int(DCD_REPORT_INTERVAL)))
            self.current_dcd_paths.append(dcd_path)

        # propagate
        prev_d1 = self.current_distance
        prev_d2 = self.current_distance2

        # live per-chunk samples for this RL action
        seg_cv1 = []
        seg_cv2 = []
        pairs = [(self.atom1_idx, self.atom2_idx), (self.atom3_idx, self.atom4_idx)]
        seg_t_ps = []

        _nan_safe_propagate(
            sim,
            int(propagation_step),
            dcdfreq=int(dcdfreq_mfpt),
            prop_index=self.step_counter,
            atom_pairs=pairs,
            out_cv1=seg_cv1,
            out_cv2=seg_cv2,
            out_times_ps=seg_t_ps,
            dt_min=MIN_STEPSIZE,
        )       

        if len(seg_cv1) > 0:
            self.episode_trajectory_segments.append(seg_cv1)
        if len(seg_cv2) > 0:
            self.episode_trajectory_segments_cv2.append(seg_cv2)

        st = sim.context.getState(getPositions=True, getVelocities=True)
        self.current_positions = st.getPositions(asNumpy=True)
        self._last_positions = self.current_positions
        self._last_topology = self.psf.topology
        self.current_velocities = None 

        d1, d2 = self._current_distances_A()
        self.current_distance = float(d1)
        self.current_distance2 = float(d2)

        # reward / termination (CV1 primary, CV2 shaping)
        delta1 = self.current_distance - prev_d1
        outward = max(0.0, delta1)
        inward = max(0.0, -delta1)

        delta2 = self.current_distance2 - prev_d2
        inward2 = max(0.0, -delta2)    # good: d2 decreases
        outward2 = max(0.0, delta2)    # bad: d2 increases

        reward = 0.0
        done = False

        in_zone1 = (TARGET_MIN <= self.current_distance <= TARGET_MAX)
        in_zone2 = (self._cv2_min <= self.current_distance2 <= self._cv2_max)

        if in_zone1 and in_zone2 and self.phase == 1:
            self.phase = 2
            self.in_zone_count = 0

        # CV2 penalty outside corridor (gentle shaping)
        if self._cv2_min is not None and self._cv2_max is not None:
            if self.current_distance2 < self._cv2_min:
                reward -= CV2_DEVIATION_PENALTY * (self._cv2_min - self.current_distance2)
            elif self.current_distance2 > self._cv2_max:
                reward -= CV2_DEVIATION_PENALTY * (self.current_distance2 - self._cv2_max)

        if self.phase == 1:
            # CV1 progress reward (unchanged)
            reward += PROGRESS_REWARD * outward

            # CV2 progress reward (toward its target center)
            reward += PROGRESS_REWARD * inward2

            # milestones on CV1 (unchanged)
            for m in DISTANCE_INCREMENTS:
                if prev_d1 < m <= self.current_distance and (m not in self.milestones_hit):
                    reward += MILESTONE_REWARD
                    self.milestones_hit.add(m)

            # milestones on CV2 (toward its target center)
            for m in DISTANCE2_INCREMENTS:
                if prev_d2 > m >= self.current_distance2 and (m not in self.milestones_hit_cv2):
                    reward += MILESTONE_REWARD
                    self.milestones_hit_cv2.add(m)

            # Treat "improvement" as either CV1 outward OR CV2 moving closer to its target
            if (outward > 0.0) or (inward2 > 0.0):
                reward += VELOCITY_BONUS + VELOCITY_BONUS
                self.no_improve_counter = 0
            else:
                reward += (BACKTRACK_PENALTY * inward) + (BACKTRACK_PENALTY * outward2)
                self.no_improve_counter += 1

            reward += STEP_PENALTY
        
        else:
            # phase 2: maximize stability inside both zones
            reward += _phase2_bowl_reward(self.current_distance, TARGET_CENTER, TARGET_ZONE_HALF_WIDTH, CENTER_GAIN)

            if self._cv2_center is not None and TARGET2_ZONE_HALF_WIDTH is not None:
                reward += _phase2_bowl_reward(self.current_distance2, self._cv2_center, TARGET2_ZONE_HALF_WIDTH, CENTER_GAIN)

            if in_zone1 and in_zone2:
                self.in_zone_count = getattr(self, "in_zone_count", 0) + 1
                reward += 0.5 * CONSISTENCY_BONUS
                if self.in_zone_count >= STABILITY_STEPS:
                    reward += 1000.0
                    done = True

            # keep original termination condition on CV1 (optional)
            if abs(self.current_distance - TARGET_CENTER) < PHASE2_TOL and in_zone2:
                reward += 1500.0
                done = True

            reward += STEP_PENALTY

        return self.get_state(), float(reward), bool(done), [self.current_distance, self.current_distance2]

# ========================= Standalone smoke-test (optional) =========================
if __name__ == "__main__":
    # This block only runs if you execute: python combined2.py
    # It is safe to import combined2.py from main.py / evaluate.py.
    print("combined2.py loaded as __main__ (2D CV smoke test)")
    env = ProteinEnvironmentRedesigned()
    s = env.reset(episode_index=1)
    print("Initial state shape:", s.shape, "CV1/CV2:", env.current_distance, env.current_distance2)
