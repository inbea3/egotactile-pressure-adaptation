# Full 63-Object Pressure-Event Inventory — 2026-11-26

## Provenance

- Dataset: `HustleHard/EgoTactile-OXT`
- Revision: `5b05f19e613758e76c330671f4c9f3c06af8af44`
- Workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45808285348
- Artifact ID: `15110822029`
- Artifact SHA-256: `5c354d9f23efd8ecb4cb9705da2824da7c28725848e5b545e9986ad6b45bc680`
- Pressure-only arrays and timestamps read; no video decoded; no model trained.

## 1. Full inventory

| Measure | Result |
|---|---:|
| Named objects | 63 |
| Direct gloved-hand participant-object recordings | 692 |
| Recordings with valid synchronized master timestamps | 692 |
| Objects with at least 10 recordings containing candidate pressure events | 63 |
| Objects with at least 7 recordings containing exactly five candidate pressure peaks | 63 |

Raw OXT metadata exposes nine labels in the 12-object/full inventory. The paper's seven high-level categories group Tools + Electronics and Sports/Exercise + Toys. These categories are descriptive only; the held-out task remains a named object.

## 2. Candidate pressure-event counts

Using 0.5-second total-load bins, three-bin smoothing, 3-second minimum peak separation, and prominence = 25% of the per-recording 95th–10th percentile load range:

- 3,543 candidate pressure events across 692 participant-object recordings.
- 596 recordings have exactly five detected peaks.
- 87 recordings have six peaks.
- 4 recordings have seven peaks.
- 1 recording has eight peaks.
- 1 recording has one peak.
- 2 recordings have zero peaks.

The pressure detector does not force five peaks per recording. These are pressure-defined events, not verified ground-truth grasp repetitions.

## 3. Prominence sensitivity

| Prominence fraction | Exactly five peaks | Six peaks | Other counts |
|---:|---:|---:|---:|
| 0.15 | 592 | 91 | 9 |
| 0.20 | 594 | 89 | 9 |
| 0.25 | 596 | 87 | 9 |
| 0.30 | 598 | 83 | 11 |

The five-peak count is stable across the tested prominence range.

## 4. Window and target feasibility

At prominence 0.25:
- 3,543 future six-region load-share targets are finite;
- 3,535 candidate events have fully covered, non-overlapping early 1 s and future 2 s windows;
- 8 events fail the non-overlap/window-coverage criterion;
- all 63 objects have at least seven participant recordings with exactly five candidate peaks, so even a strict exact-five-recording sensitivity subset remains count-feasible for 5-shot Support plus one Query.

The early-window six-region shares are being added to the full report for the subsequent no-model Support test; the first full inventory artifact contains future targets and event/window diagnostics.

## 5. Interpretation and next decision

**Stage 1 data feasibility: GO.**

The full dataset supports a participant-level 1/2/5-shot design under both:
- an inclusive pressure-event policy (retain all valid detected events, average within participant);
- a strict sensitivity policy (use only participant recordings with exactly five detected pressure peaks).

The primary analysis should not silently force five events. The inclusive policy retains the observed pressure structure; exact-five recordings provide a predeclared sensitivity subset.

Next: complete early-feature extraction, then run Stage 2A within-object versus between-object repeatability and Stage 2B leave-one-object-out no-model Support correction. No MLP/Reptile training before those tests pass.
