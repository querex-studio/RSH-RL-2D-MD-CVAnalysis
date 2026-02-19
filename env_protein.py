# env_protein.py ===============================================================
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
    try:
        from simtk.openmm.app import DCDReporter  # type: ignore
    except Exception:
        DCDReporter = None

import config


# ------------------------- small helpers --------------------------------------


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def read_params(filename: str):
    par_files = []
    with open(filename, "r") as f:
        for line in f:
            line = line.split("!")[0].strip()
            if line:
                par_files.append(line)
    return CharmmPsfFile, par_files  # not used directly here; kept for parity


def load_charmm_params(filename: str):
    par_files = []
    with open(filename, "r") as f:
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
    if skip_indices is None:
        skip_indices = set()
    else:
        skip_indices = set(skip_indices)

    force = openmm.CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", strength)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for i, pos in enumerate(pdb.positions):
        if i in skip_indices:
            continue
        if psf.atom_list[i].name in ("N", "CA", "C"):
            xyz = pos.value_in_unit(unit.nanometer)
            force.addParticle(i, xyz)

    system.addForce(force)
    return force


def propagate_protein(simulation,
                      steps: int,
                      dcdfreq: int,
                      prop_index: int,
                      atom1_idx: int,
                      atom2_idx: int):
    """
    Run MD propagation with NaN checks and simple adaptive time-stepping.

    Crucially, we now use simulation.step(dcdfreq) so DCDReporter gets frames.
    """
    from openmm import unit as u

    ctx = simulation.context
    integ = simulation.integrator

    def _positions_finite():
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True)
        arr = np.asarray(pos.value_in_unit(u.nanometer))
        return np.isfinite(arr).all()

    def _current_distance_A():
        state = ctx.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        p1 = pos[atom1_idx].value_in_unit(u.nanometer)
        p2 = pos[atom2_idx].value_in_unit(u.nanometer)
        d_nm = np.linalg.norm(p2 - p1)
        return float(d_nm * 10.0)

    # local minimization + velocity reinit
    try:
        openmm.LocalEnergyMinimizer.minimize(
            ctx, 10.0 * u.kilojoule_per_mole, 200
        )
    except Exception:
        pass
    try:
        ctx.setVelocitiesToTemperature(config.T)
    except Exception:
        pass

    n_chunks = max(1, int(steps // dcdfreq))
    distances = []
    times_ps = []
    t_ps = 0.0

    try:
        orig_dt = integ.getStepSize()
    except Exception:
        orig_dt = config.stepsize

    retries_left = int(getattr(config, "MAX_INTEGRATOR_RETRIES", 2))

    for _ in tqdm(
        range(n_chunks),
        desc=f"MD Step {prop_index:>2d}",
        colour="red",
        ncols=80,
    ):
        try:
            # IMPORTANT: use simulation.step so reporters fire
            simulation.step(dcdfreq)
            if not _positions_finite():
                raise openmm.OpenMMException("NaN positions detected")
        except Exception:
            if retries_left <= 0:
                break
            retries_left -= 1
            # reduce dt
            try:
                new_dt = max(
                    float(orig_dt) / 2.0,
                    float(
                        getattr(
                            config,
                            "MIN_STEPSIZE",
                            0.0005 * u.picoseconds,
                        )
                    ),
                )
                integ.setStepSize(new_dt)
            except Exception:
                pass
            # reinit velocities
            try:
                ctx.setVelocitiesToTemperature(config.T)
            except Exception:
                pass
            # retry this chunk
            try:
                simulation.step(dcdfreq)
                if not _positions_finite():
                    raise openmm.OpenMMException(
                        "NaN positions after recovery"
                    )
            except Exception:
                break

        distances.append(_current_distance_A())
        t_ps += float(dcdfreq) * float(
            orig_dt.value_in_unit(u.picoseconds)
        )
        times_ps.append(t_ps)

    # restore step size
    try:
        integ.setStepSize(orig_dt)
    except Exception:
        pass

    return np.array(distances, dtype=np.float32), np.array(
        times_ps, dtype=np.float32
    )


def _phase2_bowl_reward(d):
    err = abs(d - config.TARGET_CENTER)
    if err >= config.TARGET_ZONE_HALF_WIDTH:
        return 0.0
    scale = 1.0 - (err / config.TARGET_ZONE_HALF_WIDTH) ** 2
    return config.CENTER_GAIN * scale


# ---------------------- main environment class --------------------------------


class ProteinEnvironmentRedesigned:
    """
    RL environment with:
      - Gaussian bias actions
      - Milestone / zone logic
      - Per-step DCD trajectories (one DCD per RL action):
            results_PPO/dcd_trajs/ep0001_s001.dcd, ep0001_s002.dcd, ...
      - runs.txt updated for compatibility with Balint's scripts
    """

    def __init__(self):
        # episode / milestone bookkeeping
        self.milestones_hit = set()
        self.in_zone_count = 0

        self.state_size = config.STATE_SIZE
        self.action_size = config.ACTION_SIZE

        # Scalars & histories
        self.current_distance = config.CURRENT_DISTANCE
        self.previous_distance = config.CURRENT_DISTANCE
        self.best_distance_ever = config.CURRENT_DISTANCE
        self.distance_history = deque(maxlen=10)

        # Logs
        self.all_biases_in_episode = []   # (type, amp_kcal, center_A, width_A)
        self.bias_log = []                # (step, amp_kcal, center_A, width_A)
        self.backstop_events = []         # (step, m_eff_A)
        self.episode_trajectory_segments = []
        self.step_counter = 0

        # Phase flags
        self.phase = 1
        self.in_zone_steps = 0
        self.no_improve_counter = 0

        # Milestone locking
        self.locked_milestone_idx = -1
        self.backstops_A = []            # Å

        # Zone confinement
        self.zone_floor_A = None         # Å (active after entering zone)
        self.zone_ceiling_A = None

        # MD state
        self.current_positions = None
        self.current_velocities = None

        # caches for last MD state
        self._last_simulation = None
        self._last_context = None
        self._last_topology = None
        self._last_positions = None
        self.simulation = None

        # Episode / DCD bookkeeping
        self.current_episode_index = None
        self.current_run_name = None
        self.current_dcd_index = 0
        self.current_dcd_paths = []

        # atom indices
        self.atom1_idx = config.ATOM1_INDEX
        self.atom2_idx = config.ATOM2_INDEX

        # set up base system
        self.setup_protein_simulation()

        # Discrete action lookup: (amplitude, width, outward_offset)
        self.action_tuples = []
        for A in config.AMP_BINS:
            for W in config.WIDTH_BINS:
                for O in config.OFFSET_BINS:
                    self.action_tuples.append((A, W, O))
        assert len(self.action_tuples) == config.ACTION_SIZE

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
            skip_indices={self.atom1_idx, self.atom2_idx},
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
        # Scalars
        self.current_distance = config.CURRENT_DISTANCE
        self.previous_distance = config.CURRENT_DISTANCE
        self.best_distance_ever = float(self.current_distance)
        self.distance_history.clear()
        self.distance_history.append(float(self.current_distance))

        # Logs
        self.all_biases_in_episode = []
        self.bias_log = []
        self.backstop_events = []
        self.episode_trajectory_segments = []
        self.in_zone_count = 0
        self.step_counter = 0

        # Phase & progress
        self.milestones_hit = set()
        self.phase = 1
        self.in_zone_steps = 0
        self.no_improve_counter = 0

        # Locks
        self.locked_milestone_idx = -1
        self.backstops_A = []

        # MD state
        if not carry_state:
            self.current_positions = None
            self.current_velocities = None

        # Zone walls cleared
        self.zone_floor_A = None
        self.zone_ceiling_A = None

        # caches cleared
        self._last_simulation = None
        self._last_context = None
        self._last_topology = None
        self._last_positions = None
        self.simulation = None

        # Episode / DCD bookkeeping
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
            runs_txt = getattr(
                config,
                "RUNS_TXT",
                os.path.join(dcd_dir, "runs.txt"),
            )
            existing = set()
            if os.path.exists(runs_txt):
                with open(runs_txt, "r") as fh:
                    existing = {ln.strip() for ln in fh if ln.strip()}
            if run_name not in existing:
                with open(runs_txt, "a") as fh:
                    fh.write(run_name + "\n")

        return self.get_state()

    # ---------- State ----------

    def get_state(self):
        self.distance_history.append(self.current_distance)
        if len(self.distance_history) >= 3:
            recent_trend = (
                self.distance_history[-1] - self.distance_history[-3]
            ) / 2.0
        else:
            recent_trend = 0.0
        stability = 0.5
        if len(self.distance_history) >= 5:
            stability = 1.0 / (
                1.0 + np.std(list(self.distance_history)[-5:])
            )

        overall = (self.current_distance - config.CURRENT_DISTANCE) / (
            config.FINAL_TARGET - config.CURRENT_DISTANCE
        )

        state = np.array(
            [
                self.current_distance / config.FINAL_TARGET,
                max(0.0, overall),
                abs(self.current_distance - config.TARGET_CENTER)
                / max(1e-6, config.TARGET_ZONE_HALF_WIDTH),
                np.clip(overall, 0.0, 1.0),
                recent_trend / 0.1,
                float(
                    config.TARGET_MIN
                    <= self.current_distance
                    <= config.TARGET_MAX
                ),
                float(self.no_improve_counter > 0),
                stability,
            ],
            dtype=np.float32,
        )
        return state

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
        cf.addGlobalParameter(sig_name, max(1e-6, width_A / 10.0))
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_backstop_force(self, system, m_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_back_{uid}"
        mname = f"m_{uid}"
        expr = f"{kname}*({mname} - r)^2*step({mname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(config.BACKSTOP_K))
        cf.addGlobalParameter(mname, m_eff_A / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_zone_upper_cap(self, system, u_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_cap_{uid}"
        uname = f"u_{uid}"
        expr = f"{kname}*(r - {uname})^2*step(r - {uname})"
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

    def _system_with_all_forces(self):
        system = openmm.XmlSerializer.deserialize(self.base_system_xml)

        # backstop locks
        if config.ENABLE_MILESTONE_LOCKS:
            for m_eff_A in self.backstops_A:
                system = self._add_backstop_force(system, m_eff_A)

        # zone caps
        if config.ZONE_CONFINEMENT:
            if self.zone_floor_A is not None:
                system = self._add_zone_lower_cap(system, self.zone_floor_A)
            if self.zone_ceiling_A is not None:
                system = self._add_zone_upper_cap(system, self.zone_ceiling_A)

        # gaussian biases
        for (_, amp, posA, widthA) in self.all_biases_in_episode:
            system = self._add_gaussian_force(system, amp, posA, widthA)

        return system

    # ---------- action → Gaussian parameters ----------

    def smart_progressive_bias(self, action: int):
        # inside target zone: minimal perturbation
        if (
            config.TARGET_MIN
            <= self.current_distance
            <= config.TARGET_MAX
        ):
            return (
                "gaussian",
                0.0,
                max(self.current_distance - 0.2, config.TARGET_MIN),
                0.3,
            )

        A_bins = config.AMP_BINS
        W_bins = config.WIDTH_BINS
        O_bins = config.OFFSET_BINS
        nW, nO = len(W_bins), len(O_bins)

        action = int(max(0, min(action, len(A_bins) * nW * nO - 1)))
        amp_idx = action // (nW * nO)
        rem = action % (nW * nO)
        width_idx = rem // nO
        off_idx = rem % nO

        base_amp = A_bins[amp_idx]
        base_width = W_bins[width_idx]
        base_offset = O_bins[off_idx]

        progress = (self.current_distance - config.CURRENT_DISTANCE) / (
            config.FINAL_TARGET - config.CURRENT_DISTANCE
        )

        amplitude = base_amp * (3.0 - 2.0 * np.clip(progress, 0.0, 1.0))
        width = base_width * (1.5 - np.clip(progress, 0.0, 1.0))

        center = self.current_distance - (base_offset + 0.3)
        center = float(np.clip(center, 0.5, self.current_distance - 0.1))

        if self.no_improve_counter >= 2:
            escalation = min(
                1.0 + 0.7 * self.no_improve_counter,
                config.MAX_ESCALATION_FACTOR,
            )
            amplitude *= escalation

        amplitude = float(
            np.clip(amplitude, config.MIN_AMP, config.MAX_AMP)
        )
        width = float(
            np.clip(width, config.MIN_WIDTH, config.MAX_WIDTH)
        )

        return ("gaussian", amplitude, center, width)

    # ---------- Step ----------

    def step(self, action_index):
        # sanitize
        action_index = int(action_index)

        # pick Gaussian according to RL action
        g_type, amp_kcal, center_A, width_A = \
            self.smart_progressive_bias(action_index)
        amp_kcal = float(min(amp_kcal, 12.0))
        width_A = float(max(width_A, 0.3))

        # log bias
        self.step_counter += 1
        self.all_biases_in_episode.append(
            (g_type, float(amp_kcal), float(center_A), float(width_A))
        )
        self.bias_log.append(
            (self.step_counter, float(amp_kcal),
             float(center_A), float(width_A))
        )

        # rebuild system with all forces
        system = self._system_with_all_forces()

        # new Simulation for this step (mode A: per-step DCD)
        integrator = openmm.LangevinIntegrator(
            config.T, config.fricCoef, config.stepsize
        )
        sim = omm_app.Simulation(self.psf.topology, system, integrator)
        if self._last_positions is not None:
            sim.context.setPositions(self._last_positions)
        else:
            sim.context.setPositions(self.pdb.positions)

        self.simulation = sim
        self._last_simulation = sim
        self._last_topology = self.psf.topology

        # Attach DCD reporter for this step
        if DCDReporter is not None and getattr(config, "DCD_SAVE", False):
            dcd_dir = getattr(
                config,
                "RESULTS_TRAJ_DIR",
                os.path.join(config.RESULTS_DIR, "dcd_trajs"),
            )
            _ensure_dir(dcd_dir)
            # run name
            if self.current_run_name is None:
                run_prefix = getattr(config, "RUN_NAME_PREFIX", "ep")
                ep_idx = (
                    self.current_episode_index
                    if self.current_episode_index is not None
                    else 0
                )
                self.current_run_name = f"{run_prefix}{ep_idx:04d}"
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
            # track
            self.current_dcd_paths.append(dcd_path)

        # propagate MD
        dists, _ = propagate_protein(
            simulation=sim,
            steps=config.propagation_step,
            dcdfreq=config.dcdfreq_mfpt,
            prop_index=self.step_counter,
            atom1_idx=self.atom1_idx,
            atom2_idx=self.atom2_idx,
        )

        # segment bookkeeping
        dists = (
            np.asarray(dists, dtype=np.float32)
            if dists is not None
            else np.array([], dtype=np.float32)
        )
        if dists.size > 0:
            self.episode_trajectory_segments.append(dists.tolist())
            self._last_positions = (
                sim.context.getState(getPositions=True).getPositions()
            )
            last_d = float(dists[-1])
            self.distance_history.extend([float(x) for x in dists])
        else:
            last_d = float(self.current_distance)

        prev_d = float(self.current_distance)
        self.previous_distance = prev_d
        self.current_distance = last_d
        self.best_distance_ever = max(
            self.best_distance_ever, self.current_distance
        )

        # reward / termination
        delta = self.current_distance - prev_d
        outward = max(0.0, delta)
        inward = max(0.0, -delta)
        reward = 0.0
        done = False
        in_zone = (
            config.TARGET_MIN <= self.current_distance <= config.TARGET_MAX
        )

        if in_zone and self.phase == 1:
            self.phase = 2
            self.in_zone_count = 0

        if self.phase == 1:
            reward += config.PROGRESS_REWARD * outward
            for m in config.DISTANCE_INCREMENTS:
                if prev_d < m <= self.current_distance and \
                        m not in self.milestones_hit:
                    reward += config.MILESTONE_REWARD
                    self.milestones_hit.add(m)
            if outward > 0.0:
                reward += config.VELOCITY_BONUS
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
                reward -= 2 * abs(
                    self.current_distance - config.TARGET_CENTER
                )
            else:
                self.in_zone_count = getattr(
                    self, "in_zone_count", 0
                ) + 1
                reward += 0.5 * config.CONSISTENCY_BONUS
                if self.in_zone_count >= config.STABILITY_STEPS:
                    reward += 1000.0
                    done = True
                if (
                    abs(self.current_distance - config.TARGET_CENTER)
                    < config.PHASE2_TOL
                ):
                    reward += 1500.0
                    done = True
            reward += config.STEP_PENALTY

        return self.get_state(), float(reward), bool(done), dists.tolist()
