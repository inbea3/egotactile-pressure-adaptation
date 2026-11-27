# Stage 3: causal-onset binned contact trajectory feasibility

**Status:** protocol and code preparation; no data result until the SSH container run. This is exploratory because the 63 objects have already informed target choice.

## Question

Does a participant-disjoint Support recording from an unseen object improve prediction of its *time-varying* regional contact pattern, beyond the Query's observed pressure and a transition learned on other objects? Stage 2B's single integrated future share did not provide convincing evidence of this.

## Data and prediction time

Use the frozen EgoTactile-OXT Stage 1 report at revision `5b05f19e613758e76c330671f4c9f3c06af8af44`. It contains pressure-only, 0.5-second six-region load bins for each direct gloved-hand recording. No RGB/video is used. Onset is the first three-sample sustained pressure threshold crossing. The primary rule requires at least two physical sensors over 5 N; the secondary sensitivity rule requires at least one. Unlike the peak detector, these onset rules do not inspect later pressure to locate the first crossing. Onset is known when the third sample crosses the threshold.

Set the prediction cutoff to the first 0.5-second bin boundary at least one second after onset. The two complete bins preceding cutoff are the early observation. The four complete bins after cutoff are the target. No bin spans the cutoff. A bin is represented by seven values: six nonnegative regional load shares if total regional load is at least 10 N, or a one-hot seventh `no_contact` value otherwise. This records contact loss without an undefined division. The 10 N threshold is fixed from the two-sensor onset rule, not chosen by prediction performance. The one-sensor sensitivity changes only onset, not target encoding.

This is a **coarsened dynamic pressure target**, not raw 137-sensor spatial reconstruction. Bin means cannot reveal within-bin order. The first-onset restriction means only one event candidate per participant-object recording; 1-shot remains one recording, not one bin.

## Eligibility and split

Require valid Stage 1 pressure/timestamp integrity, onset and all six required bins with finite regional loads, and at least six eligible participants for a held-out object. Report exclusions by reason and object. If a participant has multiple eligible recordings for one object, average within that participant before forming Support/Query. Each object is held out from global training in turn. Each Query participant is excluded from the global transition estimate across all other objects and from Support on the held-out object. Enumerate every 1-, 2-, and 5-participant Support subset. Average subsets within Query, Query participants within object, then objects equally.

## Comparators and uncertainty

The 28-dimensional future target is four seven-dimensional simplex rows. Persistence repeats the last observed early row. The participant-blocked global transition is the object-equal mean of `future - persistence` from other objects. Support-only transition uses the same residual from Support recordings. Fixed shrinkage uses 75% blocked global and 25% Support transition. Project each predicted row to the simplex. Also report the historical global transition that retains Query-participant overlap and a same-participant transition from other objects as diagnostics. MAE averages over all 28 target values.

Report paired blocked-global minus adapted MAE, with whole-object bootstrap intervals (5,000 resamples) and improved-object counts. Compare an object residual template from other participants on the held-out object against the Query-participant-blocked global residual template and a same-participant cross-object template. No cycles, bins, Support combinations, or participants are independent bootstrap units.

Because Stage 2B found a strong participant effect, add a **known-participant diagnostic**: subtract each Support participant's own other-object transition from that participant's held-out-object transition. Add the mean centered object effect to the Query participant's other-object transition, both unshrunk and with the fixed 0.25 weight. Compare the latter to the Query participant's other-object transition alone. This uses prior data from the Query participant and therefore answers a distinct known-operator scenario; it is not evidence about a completely new participant.

## Decision for a model pilot

A bounded MLP/Reptile pilot is warranted only if at least 50 objects remain eligible, the 5-shot fixed-shrinkage gain over blocked global is positive with an object-bootstrap 95% interval above zero for both onset rules, and the object residual template beats the blocked global template under both onset rules. These criteria govern resource allocation, not a confirmatory claim. If they fail, inspect failure modes and redefine the target before training a full meta learner. If they pass, freeze object/participant splits and compare persistence, global MLP, MLP plus 1/2/5-shot fine tuning, and Reptile with matched architecture, inner steps, Support/Query data, and compute budget. No Query data may select hyperparameters. Reserve separate objects for selection versus final reporting.

The Stage 1 report's 0.5-second bins are computed from entire recordings but each input bin contains only pre-cutoff pressure; the causal-onset claim applies to the detector and input/target boundary. A later raw-stream implementation must verify deployment latency and within-bin fidelity.

