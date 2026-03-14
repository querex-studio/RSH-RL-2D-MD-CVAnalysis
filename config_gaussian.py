import os, time

# ---- Files (Not needed for analytic, but keeping variable names for safety)
psf_file = None
pdb_file = None
toppar_file = None

# PSP
START_X, START_Y = 5.0, 4.0
TARGET_X, TARGET_Y = 1.0, 1.5
TARGET_RADIUS = 0.2

# ---- Atoms ----
ATOM1_INDEX = 0
ATOM2_INDEX = 1

# ---- Targets ----
# Distance from origin
CURRENT_DISTANCE = 0.5 # Start slightly away from 0 to avoid singularities if any, though 0 is fine
FINAL_TARGET = 5.0
TARGET_CENTER = FINAL_TARGET
TARGET_ZONE_HALF_WIDTH = 1.0
TARGET_MIN = TARGET_CENTER - TARGET_ZONE_HALF_WIDTH
TARGET_MAX = TARGET_CENTER + TARGET_ZONE_HALF_WIDTH

# ---- Milestones ----
DISTANCE_INCREMENTS = [1, 2, 3, 4, 5]

# ---- Locks / confinement ----
ENABLE_MILESTONE_LOCKS = False         
LOCK_MARGIN = 0.15
BACKSTOP_K = 1000.0 # Force constant for analytical

PERSIST_LOCKS_ACROSS_EPISODES = True
CARRY_STATE_ACROSS_EPISODES = False
"""
If CARRY_STATE_ACROSS_EPISODES = True, the particle position is not reset at the start of new episode (unless done or max steps reached). 
This allows the agent to "learn through" the milestone locks and experience the reward shaping more continuously, 
    rather than always starting fresh outside the first milestone zone. It also allows for a more natural 
    learning progression where the agent can build on its previous episode's final state.
"""

FREEZE_EXPLORATION_AT_ZONE = False

ZONE_CONFINEMENT = True  # Only engages after the particle has first entered the target zone.
ZONE_K = 1000.0
ZONE_MARGIN_LOW = 0.1
ZONE_MARGIN_HIGH = 0.1
SEED_ZONE_CAP_IF_BEST_IN_ZONE = True

# ---- Observation/action ----
STATE_SIZE = 8
AMP_BINS = [0.0, 2.0, 4.0, 6.0, 8.0] # Reduced amplitude by 8x total
WIDTH_BINS = [0.2, 0.5, 0.8, 1.1]
OFFSET_BINS = [0.8, 1.6, 2.4, 3.2]
ACTION_SIZE = len(AMP_BINS) * len(WIDTH_BINS) * len(OFFSET_BINS)

# Indices into state vector (env_gaussian_2d.get_state)
STATE_IDX_IN_ZONE = 3

# ---- Bias placement strategy ----
# "current_position": metadynamics-style deposition at current particle position
# "away_from_target": previous heuristic (place hill away from target to push toward it)
BIAS_PLACEMENT_MODE = "current_position"
ENABLE_BIAS = True  # Set False to disable hill deposition and run bare-potential dynamics.

MIN_AMP, MAX_AMP = 0.0, 30.0
MIN_WIDTH, MAX_WIDTH = 0.2, 2.5
MAX_ESCALATION_FACTOR = 1.5
IN_ZONE_MAX_AMP = 1e9

# ===================== PPO ================================
N_STEPS = 8 # Longer collection
BATCH_SIZE = 8
N_EPOCHS = 6
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
LR = 3e-4
PPO_TARGET_KL = 0.03

# ===================== Episode & MD =======================
MAX_ACTIONS_PER_EPISODE = 100
SIM_STEPS = 10 # Steps per action
DT = 0.01
TEMPERATURE = 1.0
FRICTION = 1.0

# ===================== Rewards ===========================
# Phase-1
PROGRESS_REWARD = 10.0       
MILESTONE_REWARD = 100.0
BACKTRACK_PENALTY = -1.0
VELOCITY_BONUS = 5.0
STEP_PENALTY = -0.1
BIAS_PENALTY = 0.01
MAX_BIASES = 100

# Phase-2
PHASE2_TOL = 0.1
CENTER_GAIN = 100.0
STABILITY_STEPS = 10
CONSISTENCY_BONUS = 20.0
ALLOW_INWARD_CORRECTION = False   # if False: do NOT apply inward bias when r > TARGET_MAX

# ===================== Curriculum / Eval =================
TRAIN_EPISODES = 5000
PROB_FRESH_START = 0.1
EVAL_EVERY = 50
N_EVAL_EPISODES = 10
EVAL_GREEDY = True
SEED = 42
SAVE_CHECKPOINT_EVERY = 50
SAVE_PLOT_EVERY = 50
SAVE_TRAJECTORY_EVERY = 1
MOVING_AVG_WINDOW = 50
PLOT_GRID_SIZE = 220
TRAJECTORY_DPI = 140
time_tag = time.strftime("%Y%m%d-%H%M%S")

# ===================== Potential =======================
# Simple well at origin
WELL_DEPTH = 10.0 # kT (since T=1.0)
WELL_WIDTH = 1.0

# Boundary (wall) stiffness used consistently in potential and force
WALL_K = 5000.0

# ===================== Outputs =================
RESULTS_DIR = "results_gaussian"
PLOTS_DIR = "plots_gaussian"
EVAL_PLOTS_DIR = os.path.join(PLOTS_DIR, "eval")
METRICS_CSV = f"{RESULTS_DIR}/training_metrics.csv"
METRICS_PLOT = f"{RESULTS_DIR}/training_dashboard.png"
SUCCESS_PLOT = f"{RESULTS_DIR}/success_rate_analysis.png"
INITIAL_POTENTIAL_PLOT = f"{PLOTS_DIR}/initial_potential.png"
CHECKPOINT_PATH = "agent_gaussian_ckpt.pt"
BEST_CHECKPOINT_PATH = "agent_gaussian_best.pt"
DCD_SAVE = False 
