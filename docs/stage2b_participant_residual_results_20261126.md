# Stage 2B result: participant control and transition residual

**Status:** exploratory, executed 2026-11-26. [Successful run](https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45825360847), source branch `experiment/stage2b-participant-residual-20261126`, dataset revision `5b05f19e613758e76c330671f4c9f3c06af8af44`. The run passed the Stage 2B unit checks and produced the full object-level JSON artifact. The permanent numeric summary is `results/stage2b_participant_residual_summary_20261126.json`.

## Main comparison

MAE is averaged across six regional shares, then participants and 63 objects equally. The paired difference is **participant-blocked global MAE minus adapted MAE**; positive favors adaptation. Each interval bootstraps whole objects 5,000 times. One shot is one participant–object recording, with valid grasp cycles nested inside it.

| Event policy | Shots | Blocked global MAE | Fixed 0.25 Support shrinkage MAE | Paired difference, 95% CI | Objects improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| All valid detected | 1 | 0.012077 | 0.012351 | -0.000274 [-0.000371, -0.000173] | 14/63 |
| All valid detected | 2 | 0.012077 | 0.012125 | -0.000048 [-0.000133, 0.000043] | 27/63 |
| All valid detected | 5 | 0.012077 | 0.011977 | +0.000100 [0.000021, 0.000181] | 35/63 |
| Exactly five valid | 1 | 0.012325 | 0.012639 | -0.000313 [-0.000414, -0.000212] | 11/63 |
| Exactly five valid | 2 | 0.012325 | 0.012407 | -0.000081 [-0.000167, 0.000007] | 26/63 |
| Exactly five valid | 5 | 0.012325 | 0.012253 | +0.000073 [-0.000003, 0.000149] | 38/63 |

Support-only transition MAE was 0.016789, 0.014715, and 0.013006 at 1/2/5 shots on the primary policy: worse than the blocked global transition at every shot count. At 5 shots, shrinkage's 0.000100 gain is about 0.83% of the blocked global MAE. Persistence was 0.012045, and the historical global transition that retains Query-participant overlap was 0.011805. Shrinkage did not beat that historical comparator. The 0.25 coefficient was inspected in Stage 2, so the small positive result is not an independent validation.

## Residual structure

For `future_share - early_share`, the leave-one-participant object template has MAE 0.012413; the participant-blocked cross-object global template has MAE 0.012091. The object-template advantage is **-0.000323** (95% CI [-0.000530, -0.000117]); it helped only 24/63 objects. The same-participant cross-object template has MAE 0.010746. Different participants on the same object have residual pairwise MAE 0.017091, while the same participant across objects has 0.015664. The exactly-five policy shows the same ordering: object-template advantage -0.000520 (95% CI [-0.000744, -0.000297]).

These measurements do not support stable, participant-invariant object-specific transition residuals in the current six-region integrated-share target. The same-participant result suggests substantial participant or glove-fit/interaction-style signal; it does not isolate which source causes it.

## Decision and limitations

The decision criterion in the Stage 2B protocol required a 5-shot gain that was stable under exactly-five-event filtering **and** an object residual-template advantage. It is not met. A full Reptile versus MLP study on this unchanged target would be hard to interpret as evidence of adaptation to an unseen object's tactile structure. We therefore stop this particular target at a documented feasibility result.

The next bounded experiment should build a genuinely dynamic target from raw pressure: e.g. six-region load or contact shares in successive fixed time bins after onset, optionally followed by 15 pairwise region coactivation values. Freeze causal onset and windowing before fitting. First repeat the participant-controlled no-model audit on that target. If a Support signal survives, compare persistence, global MLP, MLP plus matched 1/2/5-shot fine tuning, and Reptile with identical encoder, support/query splits, adaptation steps, compute budget, and object-level intervals. In addition to held-out objects, hold out Query participants across training objects; ideally add a fully participant-disjoint sensitivity split. Use separate objects for hyperparameter selection and final reporting.

All current 63 objects were seen during exploratory design, and the event peaks can use future recording context. Current results are offline and oracle aligned; they do not establish prospective streaming robot behavior. The full object-level artifact has a 30-day GitHub Actions retention period, while the numeric summary and code remain in the repository.

