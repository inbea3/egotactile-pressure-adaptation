# Stage 5: MAML feasibility on held-out-object pressure trajectories

**Status: superseded by the longer-trained MLP comparison.** The 5-shot exact MAML advantage below applies to the original 1,500-step MLP only. A 5,500-step MLP with approximately matched training-record exposures reaches MAE **0.6247**, beating exact MAML's **0.6407**. See `docs/stage5_budget_matched_mlp_results_20261127.md` and `results/stage5_mlp_budget5500_summary.json`. The same 63 EgoTactile-OXT objects had already informed the Stage 3/4 target and model choices; none of these comparisons is confirmatory.

## Experiment

Reuse the frozen Stage 1 report, causal two-sensor onset, Stage 4 `log_region_change` target, five held-out-object folds, and identical deterministic Support subsets. Input is two observed 0.5-second bins × six `log1p` regional loads. Output is four future bins × six log-load changes. A shot is one complete participant–object record, and Support and Query on each object use distinct participants. The test scenario has a new object but known participants. Lower MAE is better.

The 12 → 32 → 32 → 24 MLP architecture is shared. Supervised MLP has 1,500 Adam steps with batch size three. First-order MAML (FOMAML) and exact second-order MAML each have 1,500 object tasks, three Support SGD steps at rate 0.05, up to three disjoint Query records, and Adam outer rate 0.001. Exact MAML differentiates through the Support updates; FOMAML copies only the adapted Query gradient. All methods use the same three-step test adaptation at rate 0.05. The meta learners have more record exposures and higher computation than the supervised MLP, so this is **not** a compute-matched comparison. Earlier optimized Reptile used 6,000 meta tasks and a post hoc test adaptation rate; it is not directly budget-matched either.

Three preselected training seeds `20261127`, `20261128`, and `20261129` were evaluated on the same 63 objects and Support combinations. For the paired intervals below, each object's metric was first averaged across seeds, then whole objects were bootstrapped 5,000 times. These intervals describe object variability conditional on the three seeds; they do not fully incorporate training-seed or dataset-selection uncertainty.

## Results

| Shots | MLP, no adaptation | MLP, adapted | FOMAML, adapted | Exact MAML, adapted |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6611 | 0.6866 | 0.6795 | 0.6767 |
| 2 | 0.6611 | 0.6721 | 0.6694 | 0.6604 |
| 5 | 0.6611 | 0.6493 | 0.6517 | **0.6407** |

At 5 shots, exact MAML improves from its own no-adaptation MAE of 0.6883 by 0.04765, with object-bootstrap 95% CI `[0.03502, 0.06185]` and 56/63 improved objects. Against adapted MLP, the paired MLP-minus-MAML improvement is **0.00862** (`[0.00323, 0.01317]`), with 50/63 improved objects. The three individual seed results for adapted MAML are 0.63575, 0.64434, and 0.64185, versus 0.64936, 0.64849, and 0.64997 for adapted MLP. Against adapted FOMAML, the paired improvement is 0.01104 (`[0.00690, 0.01495]`).

At 2 shots, adapted exact MAML improves on adapted MLP by 0.01171 (`[0.00632, 0.01646]`) but is essentially tied with unadapted MLP: 0.6604 versus 0.6611, paired CI for their difference `[-0.01099, 0.01394]`. At 1 shot, adapted exact MAML improves on adapted MLP, but unadapted MLP remains better by 0.01559 (`[0.00308, 0.02720]` in MLP's favor). Fine tuning one Support record is therefore not yet useful against the strongest simple baseline.

FOMAML gains 0.02751 from its own adaptation at 5 shots (`[0.01336, 0.04239]`), but its adapted result of 0.6517 does not beat adapted MLP's 0.6493; paired MLP-minus-FOMAML CI is `[-0.00592, 0.00087]`. The distinction between exact and first-order gradients appears material in this configuration, but that comparison also requires independent confirmation.

## Checks and interpretation

- The smoke run and two Stage 5 tests passed on AutoDL. The tests check disjoint Support/Query participants and a nontrivial first-order meta update. The repository's minimal CI lacks PyTorch, so it skips these two dependency-specific tests; both CI workflows pass.
- The MLP per-object metrics exactly match the previously saved Stage 4 MLP run for every seed, shot count, and object. All exact MAML runs returned finite metrics. No test object enters meta-training for its fold.
- The evidence supports a **5-shot adaptation advantage for continuous regional pressure-change prediction on this inspected dataset**. It does not show tactile sensor-relation or graph-edge prediction, and it does not demonstrate robot online adaptation. The data are offline human glove pressure, coarsened to six regions and 0.5-second bins.
- The current AutoDL repository contains the Stage 1 summary report, not the raw 137-sensor pressure streams. The report's time bins contain regional means, total mean, and active-sensor counts. They cannot recover within-bin temporal order or label sensor-to-sensor coordination edges.

## Revised next decision

Treat the longer-trained MLP result as the stronger current baseline. Do not claim a MAML superiority result from this Stage 5 pilot. Freeze the current comparisons and investigate a well-defined new tactile task and independent object set. If the research claim is specifically about *协同关系*, first obtain raw sensor streams, define an observable future edge target and negative controls, test its reproducibility across participants for each object, and only then train a graph meta learner. Avoid deriving both graph edges and Query labels from the same pressure window.

Numeric summaries: `results/stage5_fomaml_summary.json` and `results/stage5_exact_maml_summary.json`. Full per-object/per-seed outputs and logs remain on AutoDL under `results/stage5_*`. Code: `scripts/run_stage5_fomaml_pilot.py` and `scripts/run_stage5_exact_maml_check.py`.

