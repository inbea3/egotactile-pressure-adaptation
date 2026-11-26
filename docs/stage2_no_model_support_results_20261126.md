# Stage 2 No-Model Support Test — Results (2026-11-26)

## Provenance

- Dataset: HustleHard/EgoTactile-OXT
- Dataset revision: 5b05f19e613758e76c330671f4c9f3c06af8af44
- Stage 1 workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45809071334
- Corrected Stage 2 workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45810885815
- Artifact ID: 15114315385
- Artifact SHA-256: c038948481fe06dc217831040d8e550e542c602875f5fd691e6c8a7e5946def7
- Bootstrap: 5,000 object-level resamples; seed 20261126.
- No neural model was trained.

## 1. Analysis unit and policies

The participant-object recording is the Support/Query unit. Pressure events are averaged within participant before inference. For each held-out object, Query participant and Support participants are disjoint. Results are reported over the 63 named objects.

Two predeclared event policies were evaluated:
1. `all_detected_events`: retain valid pressure-defined events and average within participant;
2. `exactly_five_events`: retain only participant-object recordings with exactly five detected peaks, as a sensitivity analysis.

All 63 objects retained at least six valid participants under both policies.

## 2. Within-object repeatability

| Policy | Within-object participant-pair MAE (95% object-bootstrap CI) | Between-object participant-pair MAE (95% object-bootstrap CI) | Objects with within < between |
|---|---:|---:|---:|
| All detected events | 0.06214 [0.05962, 0.06487] | 0.07149 [0.06983, 0.07324] | 54/63 |
| Exactly five events | 0.06316 [0.06038, 0.06624] | 0.07241 [0.07066, 0.07425] | 56/63 |

This supports a measurable object-conditioned structure in future six-region load-share vectors, although within-object variation remains substantial. Same-participant cross-object distances are lower than unrelated cross-object distances, so participant identity is a material nuisance factor that must be controlled in the next split design.

## 3. Main no-model transition test

The current Support correction estimates a participant-specific mean transition (future share minus early share) from Support participants and applies it to the Query participant's early share. It is compared with:
- a leave-one-object-out global transition;
- early-share persistence.

### All detected events

| Shot | Support correction MAE (95% CI) | Global transition MAE (95% CI) | Persistence MAE (95% CI) |
|---:|---:|---:|---:|
| 1 | 0.01679 [0.01568, 0.01798] | 0.01181 [0.01105, 0.01261] | 0.01205 [0.01123, 0.01285] |
| 2 | 0.01471 [0.01370, 0.01575] | 0.01181 [0.01106, 0.01263] | 0.01205 [0.01126, 0.01285] |
| 5 | 0.01301 [0.01207, 0.01390] | 0.01181 [0.01106, 0.01261] | 0.01205 [0.01129, 0.01283] |

The full Support transition correction loses to both global transition and persistence at all three shot counts. Its 5-shot performance is better than its 1-shot performance, but it does not establish a positive adaptation benefit.

### Exactly five events

| Shot | Support correction MAE (95% CI) | Global transition MAE (95% CI) | Persistence MAE (95% CI) |
|---:|---:|---:|---:|
| 1 | 0.01720 [0.01598, 0.01848] | 0.01201 [0.01121, 0.01286] | 0.01218 [0.01131, 0.01303] |
| 2 | 0.01505 [0.01398, 0.01615] | 0.01201 [0.01123, 0.01289] | 0.01218 [0.01134, 0.01303] |
| 5 | 0.01331 [0.01231, 0.01428] | 0.01201 [0.01122, 0.01287] | 0.01218 [0.01137, 0.01302] |

The strict exact-five sensitivity reproduces the same conclusion.

## 4. Direct Support prototype and shrinkage diagnostics

A direct Support future-share prototype is more competitive at 5-shot:
- all-event policy: Support prototype MAE 0.04875 [0.04674, 0.05096], versus global future prototype 0.05112 [0.04891, 0.05344];
- exact-five policy: Support prototype MAE 0.04957 [0.04743, 0.05203], versus global future prototype 0.05178 [0.04945, 0.05421].

This is a separate, unconditional prototype baseline, not the same estimator as the early-conditioned transition correction. It suggests Support contains object-specific information, but does not yet demonstrate that an adaptation procedure uses that information effectively.

A fixed shrinkage sensitivity was also run for the transition correction. For 5-shot all-events, lambda=0.25 gives MAE 0.01177 [0.01100, 0.01260], versus global-only lambda=0 MAE 0.01181 [0.01106, 0.01261]. The paired object-level improvement CI crosses zero. Exact-five lambda=0.25 is similarly indistinguishable from global-only. Thus no reliable positive adaptation gain has yet been established.

## 5. Critical interpretation and Stage 2 decision

**Data feasibility: GO. Evidence for useful adaptation: NOT YET. Do not start the headline Reptile comparison yet.**

The results indicate:
1. pressure-derived future regional load shares carry object-conditioned structure;
2. direct Support prototypes at 5-shot contain some object-specific signal;
3. the current Support-estimated transition is noisier than the global transition and persistence;
4. participant identity influences structure and must be explicitly blocked or controlled;
5. shrinkage does not yet produce a statistically clear gain over global-only prediction.

## 6. Required Stage 2B correction before model training

Before training MLP/Reptile, implement and run:
- participant-blocked global baseline: no Query participant's recordings from other objects may enter global transition estimation;
- Support prototype / residual correction comparisons with identical Support/Query participant splits;
- nested selection of shrinkage using Support only, never Query;
- report paired object-level differences and confidence intervals;
- confirm the 1/2/5-shot trends under both event policies.

Only if a leakage-controlled Support method shows a reproducible improvement over global and persistence baselines should the model stage proceed. If not, revise the scientific question or representation before expensive training.
