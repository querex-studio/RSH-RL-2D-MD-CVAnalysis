import os
import time
import openmm
from openmm import unit

SEED = 42
ROOT_DIR = os.path.dirname(__file__)


def get_best_platform():
    names = [openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())]
    print(f"Available OpenMM platforms: {names}")
    if 'CUDA' in names:
        print("Using CUDA platform (GPU)")
        return openmm.Platform.getPlatformByName('CUDA')
    if 'OpenCL' in names:
        print("Using OpenCL platform")
        return openmm.Platform.getPlatformByName('OpenCL')
    print("Using CPU platform")
    return openmm.Platform.getPlatformByName('CPU')


platform = get_best_platform()

# ---- Files
psf_file = 'step3_input.psf'
pdb_file = 'traj_0.restart.pdb'
toppar_file = 'toppar.str'

# ---- Atoms
ATOM1_INDEX = 7799   # Phosphate (CV1)
ATOM2_INDEX = 7840   # Magnesium (CV1)
CV1_LABEL = "CV1 (Phosphate-Mg distance)"

# CV2: Glutamine-Asparagine (Gln43-Asn26)
# User can easily change these indices here
ATOM3_INDEX = 660    # Gln43 CA
ATOM4_INDEX = 381    # Asn26 CA
CV2_LABEL = "CV2 (Gln43 CA - Asn26 CA distance)"

# Optional multi-pair training: list of tuples; leave empty to disable
ATOM_PAIRS = []

# ---- Targets
# CV1 (Phosphate-Magnesium): Short -> Long
# Start ~3.3A, Target ~7.5A
CURRENT_DISTANCE = 3.3
FINAL_TARGET = 7.5

# CV2 (Gln-Asn): Long -> Short
# Start ~13.5A, Target ~4.5A (Contact)
CURRENT_DISTANCE_2 = 13.5
FINAL_TARGET_2 = 4.5 
TARGET_CENTER = FINAL_TARGET
TARGET_ZONE_HALF_WIDTH = 0.35
TARGET_MIN = TARGET_CENTER - TARGET_ZONE_HALF_WIDTH
TARGET_MAX = TARGET_CENTER + TARGET_ZONE_HALF_WIDTH

# ---- Milestones
DISTANCE_INCREMENTS = [3.5, 3.8, 4.2, 5.0, 6.0, 7.0]

# ---- Locks / confinement
ENABLE_MILESTONE_LOCKS = False         # final training: no hard locks
LOCK_MARGIN = 0.15
BACKSTOP_K = 3.0e4

PERSIST_LOCKS_ACROSS_EPISODES = True
CARRY_STATE_ACROSS_EPISODES = True

FREEZE_EXPLORATION_AT_ZONE = False  # do not freeze exploration

ZONE_CONFINEMENT = True
ZONE_K = 8.0e4
ZONE_MARGIN_LOW = 0.05
ZONE_MARGIN_HIGH = 0.05
SEED_ZONE_CAP_IF_BEST_IN_ZONE = True

# ---- Observation/action
STATE_SIZE = 8
AMP_BINS = [0.0, 4.0, 8.0, 12.0, 16.0]
WIDTH_BINS = [0.3, 0.5, 0.7, 1.0]
OFFSET_BINS = [0.1, 0.2, 0.5, 1.0, 1.5]      # added 0.1 for finer Phase-2
ACTION_SIZE = len(AMP_BINS) * len(WIDTH_BINS) * len(OFFSET_BINS)

MIN_AMP, MAX_AMP = 0.0, 40.0
MIN_WIDTH, MAX_WIDTH = 0.1, 2.5
MAX_ESCALATION_FACTOR = 1.5
IN_ZONE_MAX_AMP = 1e9               # no mask in zone for final training

# ===================== PPO ================================
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

# ===================== Episode & MD =======================
MAX_ACTIONS_PER_EPISODE = 32
stepsize = 0.001 * unit.picoseconds      # safer integrator
fricCoef = 2.0 / unit.picoseconds
# --- thermostat temperature ---
T = 300 * unit.kelvin

propagation_step = 5000                    # total integrator steps per action
dcdfreq_mfpt = 40                          # save interval

# NaN recovery
MAX_INTEGRATOR_RETRIES = 2
MIN_STEPSIZE = 0.0005 * unit.picoseconds

# ===================== Rewards ===========================
# Phase-1
PROGRESS_REWARD = 120.0       # per Å outward this step
MILESTONE_REWARD = 200.0
BACKTRACK_PENALTY = -15.0
VELOCITY_BONUS = 10.0
STEP_PENALTY = -0.5
# Phase-2
PHASE2_TOL = 0.08
CENTER_GAIN = 400.0
STABILITY_STEPS = 6
CONSISTENCY_BONUS = 50.0

# ===================== Curriculum / Eval =================
PROB_FRESH_START = 0.5
EVAL_EVERY = 5
N_EVAL_EPISODES = 3
SAVE_CHECKPOINT_EVERY = 5
RESULTS_DIR = os.path.join(ROOT_DIR, "results_PPO")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_CSV = os.path.join(RESULTS_DIR, "training_metrics.csv")

EVAL_GREEDY = True

time_tag = time.strftime("%Y%m%d-%H%M%S")
print(f"Start {CURRENT_DISTANCE:.2f} Å → Target {FINAL_TARGET:.2f} Å; "
      f"Zone [{TARGET_MIN:.2f}, {TARGET_MAX:.2f}] Å")
print(f"Actions: {ACTION_SIZE} (A×W×Δ); Locks: {ENABLE_MILESTONE_LOCKS}, "
      f"K={BACKSTOP_K}, margin={LOCK_MARGIN} Å")
print(f"Zone confinement: {ZONE_CONFINEMENT} (K={ZONE_K}, "
      f"margins={ZONE_MARGIN_LOW}/{ZONE_MARGIN_HIGH} Å)")

# --- nonbonded and restraints ---
nonbondedCutoff = 1.0 * unit.nanometer
backbone_constraint_strength = 100

# ===================== DCD / trajectory output =======================
# Per-episode/step DCD trajectories for external monitoring scripts
DCD_SAVE = True
RESULTS_TRAJ_DIR = os.path.join(RESULTS_DIR, "dcd_trajs")

# DCD reporter sampling interval (in MD steps) — default: match MFPT sampling
DCD_REPORT_INTERVAL = dcdfreq_mfpt

# Run naming for compatibility with original monitoring scripts
RUN_NAME_PREFIX = "ep"   # produces ep0001, ep0002, ...
RUNS_TXT = os.path.join(RESULTS_TRAJ_DIR, "runs.txt")

# Analysis runs (PCA/post-process)
RUNS_DIR = os.path.join(RESULTS_DIR, "analysis_runs")
EPISODE_PDB_DIR = os.path.join(RESULTS_DIR, "episode_pdbs")
BIAS_PROFILE_DIR = os.path.join(RESULTS_DIR, "bias_profiles")
