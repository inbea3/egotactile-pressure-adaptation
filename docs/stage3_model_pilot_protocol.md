# Bounded MLP–Reptile pilot on the dynamic pressure target

**Status:** exploratory addendum to Stage 3. The original no-model gate was not met: a five-recording fixed shrinkage correction improved MAE under both onset rules, but the raw object residual template did not beat the global template. This pilot asks whether a learned representation changes that conclusion. Its result cannot be presented as confirmatory evidence for meta learning.

## Prediction and scenario

Use only the Stage 3 primary onset rule (`two_sensors_gt_5N_3samples`) and its two observed/four future half-second, seven-channel contact rows. The seventh `no_contact` channel is retained for schema consistency, although the first full-data Stage 3 run found it was zero in all eligible early and future bins. One shot is one participant–object recording. The model input is the 14 early values; the target is the 28-dimensional future-minus-last-early residual. MAE of simplex-projected future predictions is the evaluation metric.

This is an **unseen-object, known-participant** pilot: all records of a test object are excluded from training, but its Query participant may appear on other training objects. Support participants on that test object never equal the Query participant. An actor-history baseline is computed from training objects only, so it has the same object boundary as the models. A fully new-participant claim needs another split.

## Frozen comparison

Assign eligible objects to five folds using a seeded permutation (`20261127`) and round-robin allocation, without target information. In each fold train on the other four folds and evaluate the held-out fold. The same support subsets and Query recordings are used for every model. For each object/Query/shot, select up to five combinations deterministically by a SHA-256 seed; enumerate shots 1, 2, and 5. Average combinations within Query, participants within object, then objects equally. Report object-level paired bootstrap intervals with 5,000 draws.

- Persistence repeats the last early contact row.
- Global transition averages training-object residuals equally.
- Actor transition averages the Query participant's residuals from training objects.
- Supervised MLP and MLP plus Support fine tuning.
- Reptile initialization before and after Support adaptation.

Both neural methods use the same two-hidden-layer, width-32 ReLU residual MLP, zero-initialized output layer, squared-error training objective scaled by 100, and three Support SGD steps at learning rate 0.05. Gradients are clipped to norm 1. Supervised MLP uses 1,500 Adam steps with batch size 32 and learning rate 0.001. Reptile uses 500 object-task outer iterations × three inner SGD steps, with object-uniform sampling, shot count sampled from 1/2/5, and an outer interpolation rate linearly decreasing from 0.1 to 0.01. The methods have equal nominal training gradient-step counts, though sample exposures differ; do not claim exact compute parity. No hyperparameter search or Query-based selection is performed in this pilot.

The primary paired comparison is MLP fine-tuned MAE minus Reptile adapted MAE (positive favors Reptile), by object. Also report whether fine tuning helps the supervised MLP and whether adapted Reptile beats the actor-history and global-transition baselines. A single seed is a functional pilot; any promising result requires repeat seeds and a separate object selection/evaluation design before a research claim.

