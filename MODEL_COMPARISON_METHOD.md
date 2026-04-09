# Model Comparison Method

## Scope

This note defines the comparison design for the 2D analytical-potential project.

The comparison will include four models:

- `ppo_biased`: the baseline PPO agent using the existing target-centric state directly.
- `adaptive_cvgen_like_2d`: a 2D, paper-faithful analogue of Adaptive CVgen implemented for this project.
- `ppo_tica_2d`: PPO with a fixed TICA encoder.
- `ppo_vampnet_2d`: PPO with a fixed VAMPNet encoder.

## Why TICA and VAMPNet are PPO encoders here

TICA and VAMPNet are better treated as state encoders for PPO than as separate adaptive-sampling controllers in this project.

Reasons:

1. The scientific question is representation quality.
   In this 2D system, TICA and VAMPNet are most naturally interpreted as alternative low-dimensional descriptions of the same transition process. Using them as PPO encoders isolates whether a better dynamical representation improves path discovery.

2. This keeps the control law fixed.
   If PPO remains the controller for the baseline, TICA, and VAMPNet variants, then the main changing factor is the state representation rather than the optimizer, rollout logic, or policy-update rule. That makes the comparison cleaner.

3. This avoids mixing two different methodological changes at once.
   Adaptive CVgen changes the sampling paradigm itself: it uses unbiased trajectory segments, trajectory-history scoring, TICA clustering, and seed reselection between rounds. If TICA or VAMPNet were also turned into separate adaptive-sampling controllers, the comparison would become harder to interpret because both the representation and the control mechanism would change.

4. This is closer to the intended role of TICA and VAMP-style methods in molecular simulation.
   TICA is a linear slow-mode representation. VAMPNet is a nonlinear representation optimized for slow dynamical structure. In MD, both are commonly used to build better kinetic coordinates. That maps naturally onto the role of an encoder.

5. This gives four distinct model families rather than redundant variants.
   The resulting set is:
   - biased PPO baseline
   - unbiased Adaptive-CVgen-like controller
   - PPO with linear slow coordinates
   - PPO with nonlinear slow coordinates

This is a stronger comparison than creating several methods that all change the control loop in overlapping ways.

## Why equal total environment steps is the right fairness criterion

The comparison will use equal total environment steps as the default training budget.

Here, an environment step means the actual stochastic dynamics updates used to move the particle in the analytical potential. This is the fairest budget because:

1. It measures physical sampling effort.
   In this project the expensive and scientifically meaningful part is the simulated dynamics, not just the number of PPO episodes or optimization updates.

2. The model families do not use the same episode structure.
   PPO uses episodes composed of multiple actions, and each action triggers several Langevin updates.
   Adaptive-CVgen-like sampling uses rounds, replicas, and trajectory segments rather than PPO episodes.
   Because of this, equal episode count would not correspond to equal sampling work.

3. Wall-clock time is too implementation-dependent.
   VAMPNet pretraining and PPO updates have different compute costs. Comparing by wall-clock time would partly measure code path overhead and hardware effects rather than sampling efficiency in the landscape.

4. Equal optimizer updates is also not meaningful across methods.
   The Adaptive-CVgen-like method does not use PPO updates at all, so matching update counts would be artificial.

For this project, the cleanest common budget is:

- total number of overdamped Langevin integration steps

This aligns best with the MD-style interpretation of the problem.

For the encoder-based models, the warmup trajectories used to fit TICA or VAMPNet should also be counted against that budget, because they consume simulated dynamics and provide extra information to those models.

## What the Adaptive-CVgen-like model means in this project

The local `Adaptive_CVgen-main` folder is not the full simulator/training codebase. It contains:

- the paper
- notebook-based analysis
- TICA projections
- ablation comparisons
- trajectory post-processing

It does not contain the full adaptive-sampling driver that generated those trajectories.

Because of that, the 2D model implemented here will be a paper-faithful analogue rather than a direct code transplant.

Its key properties will be:

- unbiased trajectory propagation on the analytical potential
- a high-dimensional 2D CV bank
- prior reward terms favoring target-relevant CVs
- dynamic penalties for oversampled regions
- a learned `alpha` balance coefficient
- TICA projection of historical samples
- clustering and seed reselection between rounds

This preserves the main scientific idea of Adaptive CVgen while adapting it to a 2D path-transition problem.

## Expected outputs per model

Each model will have its own result directory and plots.

Each model should produce:

- training or round metrics
- success-rate curves
- final-distance curves
- unbiased potential plot
- biased potential plot when bias exists
- two best successful full transition trajectories from start to target

Cross-model comparison plots should also be generated from the per-model outputs.
