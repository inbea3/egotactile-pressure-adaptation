# Exploratory MLP–Reptile pilot on log regional pressure change

**Status:** post hoc target change prompted by the Stage 4 object-signal audit. The same 63 objects have already informed target choice and model design; this is a functional sensitivity test, not an untouched evaluation.

## Prediction

Reuse the frozen Stage 1 inventory and Stage 4 primary two-sensor causal onset. A recording supplies two observed half-second bins of `log1p` six-region mean load (N), shape 2 × 6. Predict the change in four following bins relative to the last observed bin, shape 4 × 6. Evaluate MAE on these 24 log-load-change values. A shot remains one participant–object recording. This target retains regional spatial information and load magnitude, but still uses coarse bins rather than raw 137-sensor arrays.

## Object split and baseline

Use the same five object folds and deterministic Support subsets as the Stage 3 pilot, seed `20261127`. Each object is Query in exactly one fold. All records from its object are excluded from model training and global/actor-history baselines. Its Query participant may have records on training objects, while Support on the test object excludes that Query participant. Select up to five identical Support combinations per object/Query/shot at 1, 2, and 5 shots. Average combinations within Query, participants within object, and objects equally. Report object-level paired bootstrap intervals with 5,000 draws.

Compare persistence (zero future change), training-object global mean change, Query participant's other-object mean change, that actor-history baseline with 25% centered Support correction, supervised MLP before/after Support fine tuning, and Reptile before/after adaptation. The fixed 25% correction is inherited from Stage 3/4 and is a strong no-model comparator.

## Fixed training configuration

Both neural methods use an identical 12 → 32 → 32 → 24 ReLU residual MLP with zero-initialized output, mean-squared-error loss on log-load change, and gradient clipping at norm 1. Support adaptation uses three SGD steps at learning rate 0.05. Supervised MLP uses Adam learning rate 0.001, batch size 3, for 1,500 steps. Reptile uses 500 object-uniform task iterations; each samples 1, 2, or 5 participant records uniformly and takes three inner SGD steps, with outer interpolation decreasing linearly from 0.1 to 0.01. MLP has about 4,500 record exposures and Reptile about 4,000; optimizer and computation are not exactly matched. Do not tune on held-out Query results.

Run three training seeds `20261127`, `20261128`, `20261129` while holding folds and Support combinations fixed. The primary paired comparison is MLP fine-tuned MAE minus Reptile adapted MAE, positive favoring Reptile. Also report each model's change after Support adaptation and comparisons to actor history plus fixed Support. CIs are conditional on a training seed; show seed variation separately. A negative result in this configuration is not proof that a tuned Reptile model cannot work. A positive result cannot be confirmatory because target and all objects have already been inspected.

