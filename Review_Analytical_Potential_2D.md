# Expert Technical Review - Analytical Potential 2D PPO

**1. Project Overview**
This project implements a toy 2D analytical potential and trains a PPO agent to place Gaussian bias hills that push a particle toward a target. The environment propagates overdamped Langevin dynamics over a bounded domain [0, 2*pi] x [0, 2*pi]. The agent chooses discrete actions (amplitude, width, offset) from bins and receives a reward based primarily on distance-to-target progress.

**2. Architecture and Design Review**
The code is split into an RL agent (`agent.py`), an environment (`env_gaussian_2d.py`), a training harness (`train_gaussian.py`), and a configuration module (`config_gaussian.py`). This separation is structurally sound, but several design decisions reduce scientific clarity and reproducibility:
- The configuration file defines many parameters and even a two-phase reward scheme that are not actually used by the environment. This breaks the contract between configuration, documentation, and implementation, and increases the risk of silent misuse.
- The bias placement strategy is partially hard-coded in the environment, which undermines the narrative of a fully learned biasing policy. This design reduces the degrees of freedom of the policy and limits interpretability of learned behavior.
- The evaluation path shares logic with training in a way that does not ensure deterministic assessment of policy quality.

**3. File-Level Analysis**

**[agent.py](F:/My_Projects/RSH-Rosta-RL/shared_projects/Claris%20-%20Analytical%20Potential%202D%20-%2020260305/agent.py)**
Role: PPO agent implementation with actor-critic networks, PPO update, observation normalization, and action masking.

Issue: In-zone action masking reads the wrong state index.
- Location: `PPOAgent._mask_logits_if_needed` (lines 95-110).
- Explanation: The code uses `s[5]` to determine `in_zone`, but the environment encodes the in-zone flag at `state[3]` (see `env_gaussian_2d.py` lines 86-93). Index 5 is the bias-count fraction, not the zone flag. This creates a logical inconsistency between environment state definition and agent logic.
- Consequences: Action masking and `IN_ZONE_MAX_AMP` constraints are applied based on bias count rather than proximity to the target, which can invert or randomize the intended policy constraints and destabilize learning.
- Suggested improvement: Replace `s[5]` with `s[3]` and consider unit tests that validate state semantics across modules.

Issue: Evaluation path still uses training behavior.
- Location: `PPOAgent.act` (lines 112-138) and `evaluate` in `train_gaussian.py` (lines 188-205).
- Explanation: `evaluate()` calls `agent.act(..., training=True)` even though the comment claims no exploration noise. With `training=True`, the code can inject exploration noise and ignores `EVAL_GREEDY` gating. This violates standard RL evaluation practice.
- Consequences: Reported evaluation metrics are noisy, non-reproducible, and not comparable to deterministic policy performance. This can hide regressions or overestimate performance due to stochasticity.
- Suggested improvement: Use `training=False` in evaluation and explicitly log whether greedy or stochastic evaluation is used.

Issue: KL early-stopping is incomplete.
- Location: `PPOAgent.update` (lines 177-229).
- Explanation: When `approx_kl > PPO_TARGET_KL`, the code breaks only the minibatch loop, not the epoch loop. PPO’s KL stopping heuristic is intended to stop further updates when policy divergence is too large.
- Consequences: Additional epochs continue to update the policy even after the KL threshold is exceeded, undermining PPO’s trust-region behavior and potentially causing policy collapse or oscillations.
- Suggested improvement: Break out of both loops, or add a flag to stop the outer epoch loop once KL exceeds the target.

Issue: Observation normalization is updated on the same batch used for value/advantage estimation.
- Location: `compute_advantages` (lines 143-175).
- Explanation: `obs_rms.update()` is called on the batch used to compute values and advantages, and again on the final next state before bootstrapping. This introduces within-batch nonstationarity in input scaling.
- Consequences: Advantage estimates can be biased by the normalization update, especially for very small batches (`N_STEPS = 8`). This can increase variance and create training instability.
- Suggested improvement: Update normalization using a larger, slower-moving buffer or update only at episode boundaries. Alternatively, normalize with fixed statistics during a PPO update.

**[env_gaussian_2d.py](F:/My_Projects/RSH-Rosta-RL/shared_projects/Claris%20-%20Analytical%20Potential%202D%20-%2020260305/env_gaussian_2d.py)**
Role: Defines the analytic potential, bias forces, state representation, and reward signal.

Issue: State definition and policy masking are inconsistent.
- Location: `get_state` (lines 57-97) defines `in_zone` at index 3; agent masking expects index 5 (agent.py lines 95-110).
- Explanation: State semantics are not consistently documented or enforced. The index mismatch is a logic bug, not merely a documentation issue.
- Consequences: Policy constraints can be applied to the wrong state feature, reducing performance and interpretability.
- Suggested improvement: Define a shared state schema (constant names or enums) and use it in both agent and environment.

Issue: Force-wall stiffness is inconsistent between energy and force.
- Location: `background_potential` (lines 139-150) uses `k_wall = 50000`; `potential_force` (lines 178-189) uses `k_wall = 5000`.
- Explanation: The force should be the gradient of the potential. Using different stiffness values breaks the physical consistency between `U` and `-grad U`.
- Consequences: The dynamics no longer correspond to any underlying potential energy surface, which breaks detailed balance and can distort both dynamical behavior and stationary distribution.
- Suggested improvement: Use a single stiffness parameter for both potential and force, or explicitly model boundary conditions (reflecting or absorbing) without energy-force inconsistency.

Issue: Boundary handling uses hard clipping, which is not physically correct for confinement.
- Location: `_clip_to_domain` (lines 320-324), invoked during each integration step (lines 281-286).
- Explanation: Clipping truncates positions at the domain boundary, effectively teleporting the particle onto the wall when it crosses. This is not equivalent to reflecting or soft-wall dynamics and does not preserve stochastic dynamics.
- Consequences: Artificial accumulation of probability mass at the boundary and nonphysical dynamics. This can bias the RL reward signals and the learned policy.
- Suggested improvement: Use reflecting boundary conditions or a consistent soft-wall potential with matching force and energy.

Issue: Bias placement is largely deterministic and reduces policy expressiveness.
- Location: `step` (lines 251-262).
- Explanation: The hill center is always placed along the vector away from the target, so the agent controls only amplitude, width, and offset distance. The direction is not learned.
- Consequences: The policy cannot explore alternative biasing strategies or respond to local potential features. This makes the RL task closer to tuning a fixed heuristic rather than learning an adaptive biasing strategy.
- Suggested improvement: Allow the action to control direction (e.g., angle bins or continuous angle), or place hills at the current position as in metadynamics and let the policy control amplitude and width.

Issue: Reward structure ignores many configured terms.
- Location: `step` (lines 297-308) vs. unused config sections in `config_gaussian.py` (lines 27-49, 86-98).
- Explanation: Only progress, step penalty, bias penalty, and a terminal bonus are used. Milestones, phase-2 stability rewards, and lock mechanics are configured but unused.
- Consequences: The documented learning objectives and reward shaping are inconsistent with the actual reward. This can mislead maintainers and complicate debugging and tuning.
- Suggested improvement: Implement the documented rewards or remove the unused configuration items and update the guide.

**[train_gaussian.py](F:/My_Projects/RSH-Rosta-RL/shared_projects/Claris%20-%20Analytical%20Potential%202D%20-%2020260305/train_gaussian.py)**
Role: Training loop, evaluation, plotting, checkpointing.

Issue: Evaluation uses training mode.
- Location: `evaluate` (lines 188-205).
- Explanation: `training=True` uses exploration noise and bypasses greedy evaluation even when `EVAL_GREEDY` is set.
- Consequences: Evaluation metrics are not deterministic and not representative of the deployed policy.
- Suggested improvement: Use `training=False` for evaluation and log evaluation settings explicitly.

Issue: Checkpointing omits optimizer and normalization state.
- Location: checkpoint save/load (lines 240-245 and 276-283).
- Explanation: Only actor and critic weights are saved. `RunningNorm` statistics and optimizer state are not stored.
- Consequences: Resuming training changes the effective input scaling and optimizer momentum, which can cause regressions or make training irreproducible.
- Suggested improvement: Save and restore optimizer state and observation normalization statistics.

**[config_gaussian.py](F:/My_Projects/RSH-Rosta-RL/shared_projects/Claris%20-%20Analytical%20Potential%202D%20-%2020260305/config_gaussian.py)**
Role: Configuration for environment, rewards, and PPO.

Issue: Large sets of parameters are unused or inconsistent with implementation.
- Location: `FINAL_TARGET`, `TARGET_MIN/MAX`, `DISTANCE_INCREMENTS`, milestone locks, phase-2 rewards (lines 20-49, 86-98).
- Explanation: The environment uses XY target coordinates and a target radius, not radial bands or milestone locks. These parameters are documented but unused.
- Consequences: Configuration drift and increased risk of incorrect scientific interpretation.
- Suggested improvement: Remove or implement unused parameters, and ensure the documentation reflects actual behavior.

Issue: Bins and bounds are inconsistent.
- Location: `AMP_BINS`, `WIDTH_BINS`, `OFFSET_BINS` vs `MIN_AMP`, `MAX_AMP`, `MIN_WIDTH`, `MAX_WIDTH` (lines 53-61).
- Explanation: The bounds are never enforced. If bins change, they can exceed intended limits without validation.
- Consequences: Uncontrolled bias amplitudes can destabilize dynamics and learning.
- Suggested improvement: Validate bins against min/max bounds at startup and enforce constraints in the environment.

**4. Technical and Scientific Issues**

Nonphysical boundary conditions and energy-force mismatch.
- The combination of inconsistent wall stiffness and clipping leads to dynamics that do not correspond to a well-defined potential. In an overdamped Langevin system, correctness of `F = -grad U` is essential for physical interpretability and correct stationary distribution. The current implementation can produce biased sampling and nonphysical trajectories.

Violation of Markov property in RL state design.
- The reward depends on progress (distance change) and uses recent history (distance trend and stability). The state includes trend and stability, but the environment still relies on trajectory history in a way that may not be fully captured by the current state features. This undermines the MDP assumption and can lead to unstable PPO updates or policy instability.

Biasing strategy does not align with standard enhanced sampling methods.
- In adaptive biasing or metadynamics, hill placement is typically based on visited CVs. Here, the hill center is deterministically offset away from the target. This is more akin to a control heuristic than a physical bias potential. The resulting bias force is not obviously interpretable as a free-energy-based bias.

**5. Configuration and Parameter Issues**

Small on-policy batch and many epochs.
- `N_STEPS = 8`, `BATCH_SIZE = 4`, `N_EPOCHS = 10` (config lines 65-73). This produces high-variance gradient estimates and heavy reuse of a tiny batch, which is a known cause of PPO instability. Increasing `N_STEPS` or reducing `N_EPOCHS` would improve stability.

Uncalibrated dynamics parameters.
- `DT = 0.01`, `TEMPERATURE = 1.0`, `FRICTION = 1.0` (config lines 75-79). These are in reduced units but are not calibrated to the potential energy scale. With large wall stiffness and strong bias forces, the integrator can become unstable or dominated by clipping. This makes the physics highly sensitive to tuning.

Action discretization is coarse for a continuous control problem.
- The bias parameters are continuous but discretized into small bins (config lines 53-55). This can reduce resolution and hinder convergence to stable biasing policies.

**6. Code Quality and Implementation Problems**

- Missing validation and cross-module checks for state semantics.
- No regression tests or unit tests for key invariants (state indices, action mapping, reward shaping).
- Evaluation comments contradict actual behavior, which undermines trust in reported metrics.
- Checkpointing is incomplete, which affects reproducibility.

**7. Recommendations**

Algorithmic improvements:
- Consider a continuous action policy for amplitude, width, and offset with bounded outputs (e.g., tanh + scaling).
- Increase rollout length (`N_STEPS`) and reduce update epochs to lower variance.

Physics and numerical improvements:
- Enforce consistency between potential and force. Use a single wall stiffness or proper reflecting boundaries.
- Replace hard clipping with a physically motivated boundary condition.
- Add diagnostics for maximum force and integrate step stability (e.g., reject steps that exceed a displacement threshold).

Structural refactoring:
- Define a shared state schema and enforce it across agent and environment.
- Move action mapping into a shared module or use a dataclass to reduce mismatches.

Configuration and documentation:
- Remove unused configuration options or implement them fully.
- Make evaluation deterministic and explicitly log evaluation mode.
- Save optimizer and normalization state in checkpoints.

Summary:
The project is a useful prototype but currently mixes heuristic control with RL, and contains multiple inconsistencies that undermine physical correctness and algorithmic stability. Fixing the state index bug, evaluation mode, and boundary dynamics should be treated as high priority. Longer term, a continuous-action policy and consistent physics would make the results more scientifically meaningful.
