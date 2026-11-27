# Stage 4: object signal in causal pressure trajectories

**Status:** exploratory, specified after observing the Stage 3 no-model and MLP–Reptile results. This audit selects a better prediction target; it is not a confirmatory test of meta learning.

## Question

Do the pressure trajectory targets contain a reproducible object-specific component after accounting for participant behavior? Can other participants' records of one object predict the held-out participant's trajectory better than that participant's history on other objects?

## Data and targets

Reuse the frozen Stage 1 full-event inventory, the Stage 3 pressure/timestamp QC, causal first-onset rules, two complete 0.5-second observed bins, and four complete future bins. One participant–object record is one sample. Use both onset rules, with the two-sensor rule primary. Group multiple eligible episodes within the same participant–object before analysis. Keep objects with at least six eligible participants.

Compare three fixed representations of the **future change from the last observed bin**:

1. Six-region load-share change (4 × 6), equivalent to the informative channels of Stage 3. The no-contact channel was constant zero in its eligible data.
2. `log1p` of six regional mean loads in newtons, then the future-minus-last-observed change (4 × 6).
3. `log1p` of total six-region mean load in newtons, then the future-minus-last-observed change (4 × 1).

The last two retain force magnitude while compressing its long upper tail. The three MAE scales are different; compare each adaptation gain only to its own target's baseline. The 0.5-second means remain coarse and do not restore the raw 137-sensor pressure array.

## Descriptive variance audit

For each target, flatten its four future rows and fit three ordinary least-squares descriptions to all eligible participant–object cells: object indicators alone, participant indicators alone, and their additive combination. Report the fraction of total squared deviation explained by each, and the **incremental object fraction after participant indicators**. Because 63 object indicators can explain noise merely through fitting, compare this fraction against 500 permutations of the response **within each participant**. This preserves participant means and missing-cell counts while breaking shared object structure. Report the permutation mean, 95th percentile, excess over the mean, and empirical upper-tail fraction. This correction was added after inspecting the first Stage 4 fits and is explicitly post hoc. With one record per cell, interaction, recording noise, and unmeasured covariates cannot be separated. These are descriptive fits, not out-of-sample adaptation scores or causal effects.

## Cross-participant prediction audit

For each held-out object and Query participant, compute that participant's mean target change on all other objects. This is the actor-history baseline. For every other participant on the held-out object, subtract their own other-object history from their held-out-object change; this gives a Support-centered object effect. Select up to five deterministic Support combinations for each Query at each of 1, 2, and 5 shots, with seed `20261127` and no Query participant in Support. Use the same selections for all targets and both onset rules. Predict the Query change using actor history plus either 25% or 100% of the mean Support effect. The 25% weight is inherited from Stage 3, not tuned here.

Average Support combinations within Query, Query participants within object, then objects equally. Report actor-history MAE and paired actor-history-minus-adapted MAE, improved-object counts, and 95% whole-object bootstrap intervals with 5,000 draws. A positive difference favors Support adaptation. Never treat bins, regions, combinations, or participants as independent bootstrap units.

## Decision

A candidate target is worth further model work only if a 5-shot fixed-quarter Support correction improves actor-history MAE with a positive object-bootstrap interval under both onset rules, and its descriptive object component is materially visible after participant effects. The audit cannot provide an untouched final test because the same 63 objects have guided earlier target choices. A subsequent model study needs a separately frozen object set or a new independent data source, plus validation objects for hyperparameters. If all candidates remain weak, seek raw high-resolution pressure and a task tied to an observable physical interaction before tuning Reptile further.

