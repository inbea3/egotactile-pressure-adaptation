# Stage 2B: participant-controlled Support information and residual audit

## Purpose and status

This is an **exploratory diagnostic** of whether participant-disjoint Support recordings from a held-out object add information beyond Query early pressure and a transition learned on other objects. All 63 EgoTactile-OXT objects have already informed the project; these are not untouched confirmation data. The frozen upstream dataset revision is `5b05f19e613758e76c330671f4c9f3c06af8af44`. This stage uses only the Stage 1 pressure-derived report and no video.

## Unit, target, and split

One shot is one participant–object recording, with its valid grasp cycles averaged inside it. The input is the Query recording's early six-region integrated-load share; the target is the corresponding future share. Its transition residual is `future - early`. Each object is held out in turn from global training; each Query participant is excluded from same-object Support. Evaluate every 1-, 2-, and 5-recording Support subset of the remaining participants, then average subsets inside the Query, Query participants inside the object, and objects equally. Never treat nested grasp cycles or Support subset combinations as independent statistical units.

Two prespecified recording policies are reported: all detected valid events, and recordings with exactly five valid events. The second is a sensitivity analysis rather than a replacement of the first.

## Comparators

- **Persistence:** project Query early share to the simplex.
- **Known-participant global transition:** learn the mean transition from other objects; retains the historical Stage 2 subject overlap and is labelled accordingly.
- **Query-participant-blocked global transition:** learn the mean transition from other objects after excluding the Query participant everywhere. This is the primary global comparator for a new-participant claim.
- **Same-participant global transition:** use the Query participant's other-object recordings, a diagnostic for known-participant deployment.
- **Support transition:** add mean Support residual to Query early share.
- **Fixed shrinkage:** add `0.75 * blocked global + 0.25 * Support residual` to Query early share. The 0.25 coefficient appeared in Stage 2 exploration; it is not newly tuned here and is not confirmatory.
- **Future prototypes:** global and Support future shares without Query early input; reported for interpretation, not a fair main comparator.

All final share predictions are projected onto the probability simplex. Loss is mean absolute error across the six regions. The primary paired comparison is blocked global MAE minus fixed-shrinkage MAE, by object; positive values favor Support. Also report blocked global minus Support-only, known-participant global minus both variants, and persistence minus fixed shrinkage.

## Structural diagnostics

Measure residual MAE between different participants on the same object and between objects for the same participant. For each Query participant, estimate an object residual template using other participants on its object, a participant residual template using that participant on other objects, and a global template using other objects with the Query participant excluded. Compare templates against the Query residual. This estimates whether object-specific transition structure survives an early-state correction; it does not establish causality or online deployability.

## Uncertainty and interpretation

Use a paired bootstrap of the 63 object-level differences (5,000 resamples, fixed seed) and report 95% percentile intervals plus the count of improved objects. No p-values or cycle-level intervals. A useful signal for a bounded model pilot would be: positive object-level mean improvement for the participant-blocked 5-shot comparator, a 95% interval excluding zero, consistency under the exactly-five sensitivity policy, and an object residual-template advantage over the blocked global template. These are decision criteria specified before this Stage 2B run, but still exploratory because earlier results shaped the question. If they fail, do not describe Reptile as validated; a small model comparison may still test whether representation learning changes the outcome.

The existing peak-finding procedure can depend on the entire recording. Therefore all current results concern **offline, oracle-aligned event prediction**. A causal onset/streaming evaluation is required before claiming online robot adaptation.

