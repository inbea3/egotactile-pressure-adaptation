# Stage 5: exploratory first-order MAML on log regional pressure change

## Question

Does an initialization trained to adapt on a few participant recordings of a new object improve future regional pressure prediction compared with supervised pretraining and its own Support fine tuning?

This is an exploratory extension after inspecting the same 63 objects in Stages 3 and 4. It cannot establish a confirmatory generalization advantage. It predicts continuous pressure trajectories, not tactile graph edges or physical sensor coupling.

## Data and evaluation

Reuse the frozen Stage 1 report, the Stage 4 primary causal onset rule, the 2 × 6 early `log1p` regional loads, the 4 × 6 future-minus-last-observed target, the same five object folds, and the same deterministic Support subsets. A shot is one participant–object record. For each held-out object, exclude **all** its records from training. For a Query participant, its record on that object is excluded from Support. Training-object records of that Query participant may appear in training, matching the established known-participant scenario.

Use exactly the Stage 4 MLP architecture, loss, supervised training, adaptation, object-level averaging, and 5,000-draw object bootstrap. Evaluate 1, 2, and 5 shots, five folds, and three seeds `20261127`, `20261128`, and `20261129` with fixed folds and Support subsets. Compare against the Stage 4 MLP and no-model baselines. Primary comparisons are (1) FOMAML before versus after Support adaptation, and (2) MLP fine tuned versus FOMAML adapted. Lower MAE is better.

## First-order MAML training

Each outer iteration samples one training object uniformly and chooses 1, 2, or 5 Support participants uniformly. It draws up to three **different** Query participants from the same object. Copy the shared 12 → 32 → 32 → 24 MLP; take three Support SGD steps at learning rate 0.05, with gradient norm clipped to 1. Compute mean query MSE at the adapted weights. Copy the clipped query gradient to the shared initialization and take an Adam step at learning rate 0.001, ignoring the derivative through Support updates. This is first-order MAML (FOMAML), not exact second-order MAML. Test adaptation uses the same three SGD steps at rate 0.05.

Run 1,500 outer iterations per fold and seed. The supervised MLP gets 1,500 Adam steps with batch size 3. FOMAML therefore has many more training-record exposures and more computation; equality of optimizer steps does **not** imply equal compute. Earlier Reptile results also used a longer post hoc optimization sensitivity. Compare outcomes descriptively and state these budget differences.

## Interpretation

A within-method adaptation gain shows that the learned initialization responds to Support records. It does not alone show better generalization than MLP, nor identify a physical object-specific relation. A cross-method claim requires the paired object-level interval, consistency across seeds, and an independently frozen object set for later confirmation. The current coarse six-region bins cannot validate sensor-to-sensor coupling edges.

