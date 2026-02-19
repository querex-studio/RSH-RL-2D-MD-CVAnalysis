# env_protein_2cv.py ===========================================================
import os
import uuid
from collections import deque

import numpy as np
from tqdm import tqdm

import openmm
import openmm.unit as unit
import openmm.app as omm_app
from openmm.app import CharmmPsfFile, PDBFile

try:
    from openmm.app import DCDReporter
except Exception:
    DCDReporter = None

import config


# ---------------------- helpers ---------------------------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_charmm_params(toppar_file):
    """
    toppar_file: a text file containing a list of CHARMM parameter files,
    one per line, optionally with comments after '!'.
    """
    par_files = []
    with open(toppar_file, "r") as f:
        for line in f:
            line = line.split("!")[0].strip()
            if line:
                par_files.append(line)
    return omm_app.CharmmParameterSet(*tuple(par_files))


def add_backbone_posres(system: openmm.System,
                        psf: omm_app.CharmmPsfFile,
                        pdb: PDBFile,
                        strength: float,
                        skip_indices=None):
    """
    Add harmonic positional restraints on backbone atoms (N, CA, C, O) and heavy
    atoms. Indices in skip_indices are excluded (e.g. the CV atoms).
    """
    if skip_indices is None:
        skip_indices = set()

    force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", float(strength))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    positions = pdb.getPositions(asNumpy=True)

    for i, atom in enumerate(psf.topology.atoms()):
        if i in skip_indices:
            continue
        # backbone atoms
        name = atom.name.strip().upper()
        if name in {"N", "CA", "C", "O"}:
            xyz = positions[i].value_in_unit(unit.nanometer)
            force.addParticle(i, xyz)

    system.addForce(force)
    return force


def propagate_protein(simulation,
                      steps: int,
                      dcdfreq: int,
                      prop_index: int,
                      atom1_idx: int,
                      atom2_idx: int,
                      atom3_idx: int,
                      atom4_idx: int):
    """
    Run MD propagation with NaN checks and simple adaptive time-stepping.

    Computes two distance CVs:
      CV1 = distance(atom1_idx, atom2_idx) in Å
      CV2 = distance(atom3_idx, atom4_idx) in Å

    Returns:
      (d1_series_A, d2_series_A, times_ps)
    """
    from openmm import unit as u

    ctx = simulation.context
    integ = simulation.integrator

    def _positions_finite():
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True)
        arr = np.asarray(pos.value_in_unit(u.nanometer))
        return np.isfinite(arr).all()

    def _distance_A(i, j):
        state = ctx.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        p1 = pos[i].value_in_unit(u.nanometer)
        p2 = pos[j].value_in_unit(u.nanometer)
        d_nm = np.linalg.norm(p2 - p1)
        return float(d_nm * 10.0)  # Å

    def _reseed_velocities():
        try:
            ctx.setVelocitiesToTemperature(config.T)
        except Exception:
            try:
                ctx.setVelocitiesToTemperature(300 * u.kelvin)
            except Exception:
                pass

    n_chunks = max(1, int(steps // dcdfreq))
    d1_series = []
    d2_series = []
    times_ps = []
    t_ps = 0.0

    try:
        orig_dt = integ.getStepSize()
    except Exception:
        orig_dt = getattr(config, "stepsize", 0.002 * u.picoseconds)

    retries_left = int(getattr(config, "MAX_INTEGRATOR_RETRIES", 2))

    for _ in tqdm(range(n_chunks), desc=f"MD Step {prop_index}", leave=False):
        try:
            simulation.step(dcdfreq)
            if not _positions_finite():
                raise openmm.OpenMMException("NaN positions detected")
        except Exception:
            # attempt recovery: shrink dt and reseed velocities
            if retries_left <= 0:
                break
            retries_left -= 1
            try:
                integ.setStepSize(orig_dt * 0.5)
            except Exception:
                pass
            _reseed_velocities()
            try:
                simulation.step(dcdfreq)
                if not _positions_finite():
                    raise openmm.OpenMMException("NaN positions after recovery")
            except Exception:
                break

        d1_series.append(_distance_A(atom1_idx, atom2_idx))
        d2_series.append(_distance_A(atom3_idx, atom4_idx))

        # time in ps
        try:
            dt_ps = float(simulation.integrator.getStepSize().value_in_unit(u.picoseconds))
        except Exception:
            dt_ps = float(getattr(config, "stepsize", 0.002 * u.picoseconds).value_in_unit(u.picoseconds))
        t_ps += float(dcdfreq) * dt_ps
        times_ps.append(t_ps)

    # restore step size
    try:
        integ.setStepSize(orig_dt)
    except Exception:
        pass

    return (np.array(d1_series, dtype=np.float32),
            np.array(d2_series, dtype=np.float32),
            np.array(times_ps, dtype=np.float32))


def _phase2_bowl_reward(d):
    err = abs(d - config.TARGET_CENTER)
    if err >= config.TARGET_ZONE_HALF_WIDTH:
        return 0.0
    scale = 1.0 - (err / config.TARGET_ZONE_HALF_WIDTH) ** 2
    return config.CENTER_GAIN * scale


# ---------------------- main environment class --------------------------------

class ProteinEnvironment2CV:
    """
    Protein RL environment extended to 2 CVs (2 distances).

    - CV1 drives the existing reward/milestone/zone logic (unchanged).
    - CV2 is added to the state and returned trajectories.

    State: np.array([cv1_norm, cv2_norm], dtype=float32)

    step(...) returns:
      state, reward, done, dists_2cv
    where dists_2cv is a list of [cv1_A, cv2_A] samples across the propagation.
    """

    def __init__(self):
        # episode / milestone bookkeeping
        self.milestones_hit = set()
        self.in_zone_count = 0

        # 2D observation: two CVs
        self.state_size = 2
        if getattr(config, 'STATE_SIZE', 2) != 2:
            print('[env_protein_2cv] Warning: config.STATE_SIZE is not 2; overriding env state_size=2.')

        self.action_size = config.ACTION_SIZE

        # phase logic
        self.phase = 1
        self.no_improve_counter = 0
        self.step_counter = 0

        # distance tracking (CV1 primary)
        self.current_distance = float(getattr(config, "CURRENT_DISTANCE", 0.0))
        self.previous_distance = float(self.current_distance)
        self.best_distance_ever = float(self.current_distance)

        self.distance_history = deque(maxlen=10)

        # distance tracking (CV2)
        self.current_distance2 = float(getattr(config, "CURRENT_DISTANCE_2", self.current_distance))
        self.previous_distance2 = float(self.current_distance2)
        self.best_distance2_ever = float(self.current_distance2)
        self.distance2_history = deque(maxlen=10)

        self.cv2_final_target = float(
            getattr(config, "FINAL_TARGET_2",
                    getattr(config, "CV2_FINAL_TARGET",
                            max(1e-6, self.current_distance2)))
        )

        # DCD bookkeeping
        self.current_episode_index = None
        self.current_run_name = None
        self.current_dcd_index = 0
        self.current_dcd_paths = []

        # atom indices
        self.atom1_idx = config.ATOM1_INDEX
        self.atom2_idx = config.ATOM2_INDEX

        # second CV atom indices
        try:
            self.atom3_idx = config.ATOM3_INDEX
            self.atom4_idx = config.ATOM4_INDEX
        except Exception as e:
            raise AttributeError(
                "For 2CV protein env, define ATOM3_INDEX and ATOM4_INDEX in config.py"
            ) from e

        # set up base system
        self.setup_protein_simulation()

        # Discrete action lookup: (amplitude, width, outward_offset)
        assert len(config.ACTION_TUPLES) == config.ACTION_SIZE

        # cached last positions for carry_state
        self._last_positions = None

    # ---------- System setup ----------

    def setup_protein_simulation(self):
        print("Setting up protein MD system...")
        self.psf = omm_app.CharmmPsfFile(config.psf_file)
        self.pdb = omm_app.PDBFile(config.pdb_file)
        self.params = load_charmm_params(config.toppar_file)

        base_system = self.psf.createSystem(
            self.params,
            nonbondedMethod=omm_app.CutoffNonPeriodic,
            nonbondedCutoff=config.nonbondedCutoff,
            constraints=None,
        )
        add_backbone_posres(
            base_system,
            self.psf,
            self.pdb,
            config.backbone_constraint_strength,
            skip_indices={self.atom1_idx, self.atom2_idx, self.atom3_idx, self.atom4_idx},
        )
        self.base_system_xml = openmm.XmlSerializer.serialize(base_system)
        print("Protein MD system setup complete.")

        # initial reset
        self.reset(seed_from_max_A=None, carry_state=False,
                   episode_index=None)

    # ---------- Persistent locks seeding ----------

    def _seed_persistent_locks(self, max_reached_A):
        if not config.ENABLE_MILESTONE_LOCKS or max_reached_A is None:
            return
        self.backstops_A = []
        self.locked_milestone_idx = -1
        for m in config.DISTANCE_INCREMENTS:
            if max_reached_A >= m:
                self.backstops_A.append(m - config.LOCK_MARGIN)
                self.locked_milestone_idx += 1

    # ---------- Reset ----------

    def reset(self,
              seed_from_max_A=None,
              carry_state=False,
              episode_index=None):
        # Scalars (CV1)
        self.current_distance = float(getattr(config, "CURRENT_DISTANCE", 0.0))
        self.previous_distance = float(self.current_distance)
        self.best_distance_ever = float(self.current_distance)
        self.distance_history.clear()
        self.distance_history.append(float(self.current_distance))

        # Scalars (CV2)
        self.current_distance2 = float(getattr(config, "CURRENT_DISTANCE_2", self.current_distance))
        self.previous_distance2 = float(self.current_distance2)
        self.best_distance2_ever = float(self.current_distance2)
        if not hasattr(self, "distance2_history"):
            self.distance2_history = deque(maxlen=10)
        self.distance2_history.clear()
        self.distance2_history.append(float(self.current_distance2))

        # Normalization for CV2 (used only for state scaling)
        self.cv2_final_target = float(
            getattr(config, "FINAL_TARGET_2",
                    getattr(config, "CV2_FINAL_TARGET",
                            max(1e-6, self.current_distance2)))
        )

        # Logs
        self.all_biases_in_episode = []
        self.bias_log = []
        self.backstop_events = []
        self.episode_trajectory_segments = []
        self.episode_trajectory_segments_2cv = []
        self.milestones_hit = set()
        self.in_zone_count = 0
        self.phase = 1
        self.no_improve_counter = 0
        self.step_counter = 0

        # DCD bookkeeping per episode
        self.current_episode_index = episode_index
        self.current_dcd_index = 0
        self.current_dcd_paths = []
        if episode_index is None:
            self.current_run_name = None

        # seed cross-episode locks
        if seed_from_max_A is not None:
            self._seed_persistent_locks(seed_from_max_A)
            if config.SEED_ZONE_CAP_IF_BEST_IN_ZONE:
                if config.TARGET_MIN <= seed_from_max_A <= config.TARGET_MAX:
                    self.zone_floor_A = (
                        config.TARGET_MIN + config.ZONE_MARGIN_LOW
                    )
                    self.zone_ceiling_A = (
                        config.TARGET_MAX - config.ZONE_MARGIN_HIGH
                    )

        # runs.txt bookkeeping
        if getattr(config, "DCD_SAVE", False) and episode_index is not None:
            dcd_dir = getattr(
                config,
                "RESULTS_TRAJ_DIR",
                os.path.join(config.RESULTS_DIR, "dcd_trajs"),
            )
            _ensure_dir(dcd_dir)
            run_prefix = getattr(config, "RUN_NAME_PREFIX", "ep")
            run_name = f"{run_prefix}{episode_index:04d}"
            self.current_run_name = run_name

            runs_path = os.path.join(dcd_dir, "runs.txt")
            # append run name once per episode
            try:
                with open(runs_path, "a") as f:
                    f.write(run_name + "\n")
            except Exception:
                pass

        # carry positions from last episode if requested
        if carry_state and self._last_positions is not None:
            self._carry_positions = self._last_positions
        else:
            self._carry_positions = None

        return self.get_state()

    # ---------- State ----------

    def get_state(self):
        """Return a strict 2D observation: [cv1, cv2] (both normalized)."""

        # CV1 scaling: reuse existing FINAL_TARGET if present, else scale by 1
        cv1_scale = float(getattr(config, "FINAL_TARGET", 1.0))
        cv1 = float(self.current_distance) / max(1e-6, cv1_scale)

        # CV2 scaling: FINAL_TARGET_2 / CV2_FINAL_TARGET / fallback stored at reset
        cv2_scale = float(getattr(self, "cv2_final_target", max(1e-6, float(self.current_distance2))))
        cv2 = float(self.current_distance2) / max(1e-6, cv2_scale)

        return np.array([cv1, cv2], dtype=np.float32)

    # ---------- Forces ----------

    def _add_gaussian_force(self, system, amplitude_kcal, center_A, width_A):
        uid = str(uuid.uuid4())[:8]
        A_name = f"A_{uid}"
        mu_name = f"mu_{uid}"
        sig_name = f"sigma_{uid}"
        expr = f"{A_name}*exp(-((r-{mu_name})^2)/(2*{sig_name}^2))"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(A_name, amplitude_kcal * 4.184)  # kcal→kJ
        cf.addGlobalParameter(mu_name, center_A / 10.0)        # Å→nm
        cf.addGlobalParameter(sig_name, width_A / 10.0)        # Å→nm
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_zone_upper_cap(self, system, u_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_cap_{uid}"
        uname = f"u_{uid}"
        expr = f"{kname}*(r-{uname})^2*step(r-{uname})"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(config.ZONE_K))
        cf.addGlobalParameter(uname, u_eff_A / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_zone_lower_cap(self, system, l_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_floor_{uid}"
        lname = f"l_{uid}"
        expr = f"{kname}*({lname} - r)^2*step({lname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(config.ZONE_K))
        cf.addGlobalParameter(lname, l_eff_A / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    # ---------- Action policy (unchanged) ----------

    def smart_progressive_bias(self, action_index):
        """
        Returns: (g_type, amplitude_kcal, center_A, width_A)

        Uses config.ACTION_TUPLES, milestone logic, etc. (unchanged from your env).
        """
        g_type, A, W, O = config.ACTION_TUPLES[action_index]

        # progressive outward center by offset
        center_A = float(self.current_distance + O)
        width_A = float(W)
        amp_kcal = float(A)

        return g_type, amp_kcal, center_A, width_A

    # ---------- Build simulation for each RL step ----------

    def _make_simulation(self, system):
        integ = openmm.LangevinIntegrator(
            config.T,
            config.friction,
            config.stepsize,
        )
        platform = openmm.Platform.getPlatformByName(config.platform_name)
        sim = omm_app.Simulation(self.psf.topology, system, integ, platform)

        # set positions
        if self._carry_positions is not None:
            sim.context.setPositions(self._carry_positions)
        else:
            sim.context.setPositions(self.pdb.positions)

        # temperature velocities
        try:
            sim.context.setVelocitiesToTemperature(config.T)
        except Exception:
            pass

        return sim

    # ---------- Step ----------

    def step(self, action_index):
        # sanitize
        action_index = int(action_index)

        # pick Gaussian according to RL action
        g_type, amp_kcal, center_A, width_A = self.smart_progressive_bias(action_index)
        amp_kcal = float(min(amp_kcal, 12.0))
        width_A = float(max(width_A, 0.3))

        # log bias
        self.step_counter += 1
        self.all_biases_in_episode.append(
            (g_type, float(amp_kcal), float(center_A), float(width_A))
        )
        self.bias_log.append(
            {
                "step": int(self.step_counter),
                "action": int(action_index),
                "type": str(g_type),
                "A_kcal": float(amp_kcal),
                "center_A": float(center_A),
                "width_A": float(width_A),
            }
        )

        # reconstruct base system
        base_system = openmm.XmlSerializer.deserialize(self.base_system_xml)

        # apply gaussian bias (CV1 only)
        system = self._add_gaussian_force(base_system, amp_kcal, center_A, width_A)

        # apply zone caps (CV1 only, unchanged)
        if getattr(config, "ENABLE_ZONE_CAPS", False):
            u_eff_A = getattr(self, "zone_ceiling_A", config.TARGET_MAX) if getattr(config, "ENABLE_MILESTONE_LOCKS", False) else config.TARGET_MAX
            l_eff_A = getattr(self, "zone_floor_A", config.TARGET_MIN) if getattr(config, "ENABLE_MILESTONE_LOCKS", False) else config.TARGET_MIN
            system = self._add_zone_upper_cap(system, u_eff_A)
            system = self._add_zone_lower_cap(system, l_eff_A)

        # create sim
        sim = self._make_simulation(system)

        # per-step DCD
        if getattr(config, "DCD_SAVE", False) and self.current_episode_index is not None:
            if DCDReporter is None:
                raise RuntimeError("DCDReporter unavailable; install OpenMM app reporters properly.")

            dcd_dir = getattr(
                config,
                "RESULTS_TRAJ_DIR",
                os.path.join(config.RESULTS_DIR, "dcd_trajs"),
            )
            _ensure_dir(dcd_dir)
            self.current_dcd_index += 1
            dcd_name = (
                f"{self.current_run_name}_s{self.current_dcd_index:03d}.dcd"
            )
            dcd_path = os.path.join(dcd_dir, dcd_name)
            interval = int(
                getattr(
                    config,
                    "DCD_REPORT_INTERVAL",
                    config.dcdfreq_mfpt,
                )
            )
            sim.reporters.append(DCDReporter(dcd_path, interval))
            self.current_dcd_paths.append(dcd_path)

        # propagate MD (2 CVs)
        d1s, d2s, _ = propagate_protein(
            sim,
            steps=int(config.sim_steps_mfpt),
            dcdfreq=int(config.dcdfreq_mfpt),
            prop_index=self.step_counter,
            atom1_idx=self.atom1_idx,
            atom2_idx=self.atom2_idx,
            atom3_idx=self.atom3_idx,
            atom4_idx=self.atom4_idx,
        )

        # segment bookkeeping (2 CVs)
        d1s = (
            np.asarray(d1s, dtype=np.float32)
            if d1s is not None
            else np.array([], dtype=np.float32)
        )
        d2s = (
            np.asarray(d2s, dtype=np.float32)
            if d2s is not None
            else np.array([], dtype=np.float32)
        )

        if d1s.size > 0 and d2s.size == d1s.size:
            dists_2cv = np.stack([d1s, d2s], axis=1)  # Å, Å
        elif d1s.size > 0 and d2s.size == 0:
            dists_2cv = np.stack([d1s, np.full_like(d1s, self.current_distance2)], axis=1)
        elif d2s.size > 0 and d1s.size == 0:
            dists_2cv = np.stack([np.full_like(d2s, self.current_distance), d2s], axis=1)
        else:
            dists_2cv = np.zeros((0, 2), dtype=np.float32)

        # keep CV1-only segments for compatibility
        if d1s.size > 0:
            self.episode_trajectory_segments.append(d1s.tolist())

        if dists_2cv.shape[0] > 0:
            self.episode_trajectory_segments_2cv.append(dists_2cv.tolist())
            self._last_positions = sim.context.getState(getPositions=True).getPositions()
            last_d1 = float(dists_2cv[-1, 0])
            last_d2 = float(dists_2cv[-1, 1])
            self.distance_history.extend([float(x) for x in d1s])
            self.distance2_history.extend([float(x) for x in d2s])
        else:
            last_d1 = float(self.current_distance)
            last_d2 = float(self.current_distance2)

        prev_d = float(self.current_distance)
        self.previous_distance = prev_d
        self.current_distance = float(last_d1)

        self.previous_distance2 = float(self.current_distance2)
        self.current_distance2 = float(last_d2)

        self.best_distance_ever = max(self.best_distance_ever, self.current_distance)
        self.best_distance2_ever = max(self.best_distance2_ever, self.current_distance2)

        # reward / termination (unchanged: CV1 only)
        delta = self.current_distance - prev_d
        outward = max(0.0, delta)
        inward = max(0.0, -delta)
        reward = 0.0
        done = False
        in_zone = (config.TARGET_MIN <= self.current_distance <= config.TARGET_MAX)

        if in_zone and self.phase == 1:
            self.phase = 2
            self.in_zone_count = 0

        if self.phase == 1:
            reward += config.PROGRESS_REWARD * outward
            for m in config.DISTANCE_INCREMENTS:
                if prev_d < m <= self.current_distance and m not in self.milestones_hit:
                    reward += config.MILESTONE_REWARD
                    self.milestones_hit.add(m)

            if self.current_distance >= config.FINAL_TARGET:
                reward += config.FINAL_REWARD
                done = True

            reward += config.STEP_PENALTY
            if inward > 0.02:
                reward += config.BACKTRACK_PENALTY
            if in_zone:
                self.phase = 2
                self.in_zone_count = 1
                reward += config.CONSISTENCY_BONUS
        else:
            reward += _phase2_bowl_reward(self.current_distance)
            if not in_zone:
                self.phase = 1
                self.in_zone_count = 0
                reward -= 2 * abs(self.current_distance - config.TARGET_CENTER)
            else:
                self.in_zone_count = getattr(self, "in_zone_count", 0) + 1
                reward += 0.5 * config.CONSISTENCY_BONUS
                if self.in_zone_count >= config.STABILITY_STEPS:
                    reward += 1000.0
                    done = True
                if abs(self.current_distance - config.TARGET_CENTER) < config.PHASE2_TOL:
                    reward += 1500.0
                    done = True
            reward += config.STEP_PENALTY

        return self.get_state(), float(reward), bool(done), dists_2cv.tolist()


# Backward-compatible alias (if older scripts import ProteinEnvironmentRedesigned)
ProteinEnvironmentRedesigned = ProteinEnvironment2CV
