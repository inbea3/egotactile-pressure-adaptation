# Frozen Experimental Plan — EgoTactile Pressure-Only Few-Shot Structure Adaptation

- Protocol version: 1.0
- Freeze date: 2026-11-26
- Dataset: HustleHard/EgoTactile-OXT
- Primary subset: directly paired `gloved_hand` only
- Modality: pressure measurements and timestamps only
- License: CC BY-NC 4.0; research use must remain non-commercial and properly attributed.

This protocol supersedes the multi-dataset candidate-selection stage. No other dataset is part of the primary study. Do not alter the original `inbea3/touch` repository.

## 1. Research question and claim boundary

**Question:** For a previously unseen named object, does a small support set of independent gloved-hand pressure trajectories improve prediction of that object's future regional pressure-load structure from an early pressure observation, and does Reptile initialization improve this adaptation over ordinary MLP fine-tuning?

The held-out unit is a **named object**, not one of the seven broad object categories. Claims must say "unseen named object" and must not say "unseen category". This is human instrumented-glove evidence, not robot hardware validation.

## 2. Data inclusion and exclusions

Include only directly paired `gloved_hand` trajectories with valid pressure vectors, timestamps, object identity, and recoverable episode boundaries.

Exclude from the primary analysis:
- all `bare_hand` weakly paired trajectories;
- corrupted or incomplete pressure/time records;
- records with unresolved object or trajectory identity;
- trajectories too short for the frozen early/future windows;
- any record whose sensor mapping cannot be verified.

Do not exclude an object based on model performance. QC warnings/failures are not silently dropped: inspect the underlying evidence, apply the same predeclared signal-integrity criteria, and publish an object/trajectory exclusion log. Report the three QC3 object-level failures (SamyangSpicyChickenCupNoodles, SprayBottle, TennisBall) separately; they are not automatic exclusions unless raw pressure/time integrity fails the criteria.

## 3. Sensor representation

Preserve three representations:
1. original 256-value pressure record and source indices;
2. OXT 162-layout mapping;
3. 137 unique physical sensor indices implied by the layout mapping.

Never treat 162 layout cells as 162 independent physical sensors. A physical sensor repeated across layout cells must be represented once in the unique-sensor analysis. Any spatial raster used for visualization is a derived representation and cannot be described as independent sensor count.

Before target construction, verify mapping JSON, index range, duplicate indices, units, and source-to-OXT equality on sampled frames. Keep a machine-readable mapping manifest and tests.

## 4. Recording and repetition units (critical clarification)

The OXT `episode_ends` unit is a **participant-object recording**, not one individual grasp repetition. The EgoTactile paper's acquisition protocol states that each participant performs five repetitions per object, each containing approach, contact, grasp/hold, release, and retreat, taking approximately six seconds; one participant-object recording therefore contains about five repeated grasp cycles over roughly 30 seconds.

For this study:
- one OXT episode = one participant-object recording;
- one recording contains approximately five nested grasp repetitions;
- one K-shot Support sample = one complete participant-object recording from one participant;
- Query must be a different participant's complete recording for the same held-out object;
- never split the five repetitions from one participant-object recording across Support and Query;
- repeated grasp cycles are nested within participant and must not be treated as independent inferential units.

Thus 1/2/5-shot means 1/2/5 distinct participant recordings, **not** 1/2/5 individual grasps. Primary inference remains at the named-object level; participant-level variability is explicitly modeled/described.

The paper describes seven high-level categories. Raw OXT metadata may use finer labels (for example, Tools vs Electronics); map these to the paper's seven categories only for descriptive stratification, never silently relabel object identities.

Source: https://arxiv.org/abs/2606.09243 (Appendix A.1.1 and acquisition protocol).

## 5. Recording inventory (Stage 0)

Build one row per participant-object recording with:
- object name and upper-level category;
- participant ID;
- acquisition mode and pairing quality;
- source repeat/sequence ID;
- episode index and start/end frame indices;
- frame count and duration;
- timestamp interval median, IQR, min/max, monotonicity;
- pressure vector shape, dtype, finite/missing fraction, min/max/quantiles;
- source archive path and SHA-256;
- eligibility flag and explicit exclusion reason.

The 5-shot gate requires at least six independent eligible trajectories for an object (five support trajectories plus one query trajectory). Report counts at object and participant levels. Do not split frames from one trajectory to manufacture independent shots.

## 6. Contact onset and windowing (Stage 1)

Use actual tactile timestamps, not an assumed constant frame rate. Initial archive inspection found repeated tactile timestamps whose adjacent pressure vectors were identical in all repeated-timestamp rows in the three sampled objects. Therefore, within each episode, collapse consecutive rows with the same tactile timestamp only when their pressure vectors are exactly identical; retain one pressure sample at that timestamp. If equal timestamps carry different pressure vectors, flag the episode and do not silently average them. Process episodes independently because timestamps may reset at episode boundaries. Report duplicate, conflicting, and negative intervals. Use zero-order-hold time weighting over the resulting unique tactile timestamps, with an explicitly documented endpoint convention. Do not sort timestamps globally.

Reconstruct the paper's per-sensor 5 N noise filter only as a candidate preprocessing rule. It is not automatically the whole-hand contact-onset detector.

Before windowing, validate timestamp semantics using the frozen duplicate rule above. Require no negative within-episode tactile intervals, no conflicting duplicate-time pressure samples, and a valid monotonic frame/tactile time relationship. Episodes failing these checks are flagged with reasons, not silently repaired.

Define and compare:
- per-sensor denoising/activation threshold;
- trajectory-level onset from aggregate pressure over valid sensors;
- continuous-contact criterion.

Freeze one onset rule after signal-only diagnostics, before examining model performance. Record the rule, thresholds, and sensitivity alternatives.

Primary windows after detected onset:
- early input: [0, 1.0) seconds;
- future target: [1.0, 3.0) seconds.

No overlap. Require valid coverage of both windows and minimum timestamp coverage. If too many trajectories fail, any window change is exploratory, must be documented, and then frozen before Stage 2. Never tune windows using prediction performance.

## 7. Target construction

Primary target: future six-region relative load-share vector (thumb, index, middle, ring, little, palm), using the official sensor mapping. Reconstruct physical sensors by grouping the 162 layout channels by source raw index: keep single-cell sensors once and sum all six virtual cells for each bend sensor (each virtual cell stores one sixth of its physical sensor value). Then sum the 137 physical sensor values within each of the six regions. For each time sample, compute regional load and total load; define regional share only when total load exceeds the valid-load threshold frozen from signal-only diagnostics. Summarize the six shares over the future window using a predeclared time-weighted aggregation.

Also retain:
- future regional contact-time proportion (using region-specific activation thresholds frozen from signal-only diagnostics);
- future absolute total load summary;
- pairwise regional co-activation as a secondary target.

A future window with insufficient aggregate load is invalid for relative-share targets. Define the minimum total-load rule from sensor noise/baseline diagnostics before model evaluation. Do not normalize near-zero load into arbitrary shares.

Pearson correlation matrices and r >= 0.7 edges are secondary sensitivity analyses only, not the primary target.

Input features must be computed only from the early window. Targets must be computed only from the future window. No future-derived information may enter normalization, feature selection, or input construction.

## 8. Stage 2A — structure repeatability

For each object, compare future target vectors across independent trajectories. Compare within-object distances against between-object distances. Repeat the analysis stratified by participant and, where possible, on participant-disjoint pairs.

Report object-level distributions and the number of objects with within-object similarity exceeding between-object similarity. Do not let high-trajectory-count objects dominate the aggregate.

This stage tests whether named-object-specific future structure is sufficiently repeatable to be a meaningful adaptation target.

## 9. Stage 2B — no-model Support information test

For each eligible held-out object, use K complete participant-object recordings from K distinct participants as Support and a different participant's complete recording as Query. Each recording retains its five nested grasp repetitions intact. Repeat valid participant-level Support/Query combinations where feasible.

Compare:
1. **Early persistence:** predict Query future structure using its own early structure.
2. **Global transition:** estimate mean future-minus-early change from other training objects only.
3. **Object Support correction:** estimate future-minus-early change from the K Support trajectories of the held-out object and apply it to the Query early structure.

Evaluate K = 1, 2, 5. The primary shot comparison uses the same intersection of objects eligible for 5-shot; 1/2-shot on larger object sets may be reported separately.

Average episodes within each object first; inferential summaries use named objects as units, never frames, sensor cells, or episode combinations as independent subjects.

Entry gate to neural models:
- within-object future structure has demonstrable repeatability;
- Support correction improves over the strongest simple baseline on a majority of eligible objects with nontrivial effect size;
- the pattern is not explained solely by one participant or a few objects.

If this gate fails, stop and revise the estimand/window/interaction definition. Do not proceed to Reptile merely to obtain a positive result.

## 10. Outer split for neural experiments

After Stage 2 passes, freeze object-disjoint train/validation/test sets. All trajectories of a named object remain in exactly one outer set. Stratify by upper-level category and participant coverage as far as the data allow, using metadata only.

Validation objects are used for model selection. Test objects remain untouched until all preprocessing, target definitions, hyperparameters, and training budgets are frozen. If the number of eligible objects is too small for a defensible holdout, use nested object-grouped cross-validation and label results exploratory; do not claim a pristine independent test.

Support and Query must always be distinct full trajectories. Add participant-disjoint sensitivity analysis where sample size permits.

## 11. Stage 3 — model comparison

Compare in this order:
1. simple Stage 2 baselines;
2. pooled MLP zero-shot;
3. pooled MLP with K-shot ordinary fine-tuning;
4. Reptile initialization with the same K-shot adaptation.

Same feature inputs, targets, object split, architecture, normalization, optimizer family, support schedule, and test-time update budget. Record any unavoidable compute/data-access mismatch. Reptile is not assumed to win.

Primary metric: per-object mean structural MAE. Secondary: regional-share MAE, contact-time error, RMSE, correlation/rank metrics where mathematically appropriate, improvement relative to early persistence, number of improved objects, object-level paired differences and uncertainty intervals, and compute/adaptation cost.

No GNN in the first model comparison. Consider graph structure only after an MLP/meta-learning comparison demonstrates a meaningful unresolved structural limitation.

## 12. Statistical unit and reporting

The independent inferential unit is the named object. First average all episodes and repeated support/query combinations within each object. Then calculate object-level summaries and paired differences. Bootstrap objects, not frames, taxels, or episodes. Explicitly report small-n limitations.

Report all seeds, failures, exclusions, per-object metrics, data checksums, code commit, configurations, and run logs. No cherry-picking objects or seeds.

## 13. Execution order and stop/go

- Gate 0: inventory, mapping, and data-integrity tests.
- Gate 1: freeze onset/windows and target definitions.
- Gate 2: repeatability and no-model Support information tests.
- Gate 3: object-disjoint neural protocol.
- Gate 4: MLP versus ordinary fine-tuning versus Reptile.
- Gate 5: sensitivity, uncertainty, and thesis claim audit.

No stage may be skipped. No neural training before Gates 0–2 pass.
