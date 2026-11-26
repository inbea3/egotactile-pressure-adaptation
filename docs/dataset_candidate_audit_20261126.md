# Candidate Dataset Audit — 2026-11-26

Status: **desk audit completed; empirical file-level audit not yet run**.

This document records claims verified against official dataset cards/project pages on 2026-11-26. Counts below are publisher-reported, not independently recomputed by this repository. No raw data has been downloaded in this audit and no experimental results are claimed.

## Executive decision

**Primary candidate: EgoTactile (gloved-hand pressure trajectories), conditional GO for a small metadata/sample audit.**

Reason: the dataset has named everyday objects, participant identifiers, source trajectory boundaries, a directly paired gloved-hand subset, and a compact object-level OXT packaging. Its pressure signal can be isolated without using video. However, the key feasibility number — how many distinct eligible gloved-hand trajectories each named object has, and how many participants contribute to each object — is not yet established. The 5-shot design must not be assumed feasible until those counts are computed.

**Secondary candidate: TouchScale, conditional / reserve.**

It has substantially more episodes and dense pressure-only signals can be extracted from the glove streams, but the currently hosted release is 100 hours, access requires accepting dataset conditions while sharing Hugging Face account information, and the full data is approximately 1.05 TB. More importantly, episode metadata's `items` field is free text and the 193 meshes are a separate object asset inventory; do not assume the 905+ episode-level objects map cleanly to stable repeated physical-object IDs. Audit metadata before considering it a primary object-held-out benchmark.

**External-domain candidate: TUM-ICS Hide-and-Seek (HAS), not a primary replacement.**

It provides 44-dimensional left robot-skin force readings and many episodes, but the task is robot object/weight classification, with 33 physical objects and 61 object-weight labels. It is a useful later cross-embodiment external-domain study only if a defensible pressure-only temporal-structure target can be defined. Never pool its channels or raw outcomes with human glove data.

## Candidate comparison

| Dataset | Verified public-card facts | Potential fit | Main blockers / risks | Current decision |
|---|---|---|---|---|
| EgoTactile / EgoTactile-OXT | 63 named object-level tasks; 767 trajectories; 314,100 aligned frames; 692 directly paired gloved-hand trajectories and 75 weakly paired bare-hand trajectories; 12 participants; 7 categories; CC BY-NC 4.0. Raw source includes a 256-value right-glove vector; OXT exposes a 162-cell layout. | Strong direct fit for named-object holdout and pressure-only within-trajectory early-to-future prediction. | Per-object/per-participant trajectory counts still unknown; only one glove side in OXT pressure stream; layout is not 162 unique sensors; 15 Hz synchronized card vs ~17 Hz project-page description; non-commercial license. | **Primary; conditional GO to sample audit.** |
| TouchScale | Current HF card says 100 h, 15,324 episodes, 929 tasks, 22 scenes, 905+ episode-level objects, 193 textured object meshes, 24 participants; 880 taxels per hand, normal force in N plus tangential components; 35–90 Hz timestamped; CC BY-NC 4.0. Full data approx. 1.05 TB. | Very strong pressure signal scale and rich repeated task/contact opportunities; normal force can be isolated without video. | File access gated by acceptance of conditions; very large data; episode `items` are free-text and do not necessarily match mesh names; 193 mesh assets are not evidence of only/ exactly 193 episode object identities; repeated physical-object identity and trial structure require auditing. | **Reserve; metadata-first audit only.** |
| TUM-ICS Hide-and-Seek | 1,120,116 frame rows; about 4.3k trajectories; 33 physical objects; 61 object-weight categories; 44-D `obs_skin_force_left`; CC BY 4.0. | Pressure-only field is explicit; robot tactile skin; can support an independent embodiment/domain comparison. | Different sensor geometry and robot embodiment; label target is object/weight classification, not the proposed human-glove future regional load structure; same physical object has weight variants that must remain grouped. | **External-domain candidate; do not mix with primary data.** |

## EgoTactile sensor representation — mandatory correction

The OXT card states that the original source contains a 256-value right-glove record. The conversion exposes the official 162-location glove layout while retaining the original 256-value vector. The 162-cell layout is a pressure topology with **137 unique physical sensor indices**; five bend sensors are each expanded to six layout cells.

Therefore:
- preserve the raw 256-vector and its source index semantics;
- preserve the exact 162-cell layout mapping;
- explicitly record the 137 unique physical indices;
- do not treat 162 layout cells as 162 independent physical sensors;
- do not average repeated layout cells until the intended physical-sensor aggregation is documented and validated;
- do not infer hand-joint labels or kinematics from this pressure topology.

The OXT card says schema prechecks passed for all 63 objects and exact source comparison covered 314,100 tactile frames and timestamps. This is a publisher/converter report, not a substitute for our own sample-level verification.

## EgoTactile inclusion policy

For the primary experiment, include only records that satisfy all of the following:
1. `acquisition_mode == gloved_hand` (or the equivalent source metadata value verified from actual files);
2. directly paired pressure recording, not weakly paired bare-hand;
3. pressure vector and timestamps present, finite, and aligned;
4. trajectory boundaries recovered from `meta/episode_ends` and checked against metadata;
5. participant, object, repeat/sequence identifiers retained where present;
6. no cross-trajectory concatenation when constructing a support/query sample.

The 75 weakly paired bare-hand records must not be mixed into the main task because the visible bare hand and off-camera instrumented hand may not have precisely matching contact geometry or pressure events.

## Time-rate and threshold caveats

The OXT card describes 15 Hz after synchronization, while the project page describes approximately 17 Hz tactile acquisition. Use timestamps from the actual released files rather than assuming a nominal fixed rate. Report observed sampling interval distribution.

The EgoTactile paper does report a per-sensor noise filter of 5 N and a 200 N scaling reference (reported as the 99.9th percentile). This supports using 5 N as a reproduction candidate for per-sensor denoising/contact state, but **does not by itself define whole-hand contact onset or prove that 5 N is optimal for our new estimand**. Our pipeline must distinguish per-sensor thresholding from trajectory-level contact-start detection, inspect raw pressure baselines and sampling intervals, and predefine any aggregation rule before constructing windows. Do not silently treat the per-sensor 5 N rule as a validated whole-hand onset detector.

## Next empirical audit (small, no full download)

1. Retrieve OXT file manifest and `supplementary_tactile_note.json`.
2. Select a small, metadata-stratified set of object archives; do not choose objects based on model performance.
3. Read only pressure arrays, timestamps, episode ends, and permitted metadata. Do not decode video.
4. Validate raw 256-vector, 162-layout mapping, unique-index count, pressure units, timestamps, episode boundaries, and finite/missing values.
5. Build one row per gloved-hand trajectory with object, participant, acquisition mode, repeat/sequence, duration, frame count, sampling interval summary, pressure quality, source archive, and checksum.
6. Produce object-level counts: total eligible trajectories, distinct participants, distinct repeats, and trajectories passing the preliminary 1 s + 2 s window rule.
7. Decide whether 1/2/5-shot is feasible on the same common set of objects. A 5-shot episode requires at least six independent eligible trajectories for that object.
8. Only after this audit freeze target construction and run the no-model feasibility experiment.

## Sources

- EgoTactile official project page: https://egotactile.github.io/
- Original EgoTactile Hugging Face dataset card: https://huggingface.co/datasets/HustleHard/EgoTactile
- EgoTactile-OXT dataset card and conversion notes: https://huggingface.co/datasets/HustleHard/EgoTactile-OXT
- TouchScale Hugging Face dataset card: https://huggingface.co/datasets/2077AIDataFoundation/TouchScale
- TouchScale paper: https://arxiv.org/abs/2610.10288
- TUM-ICS Hide-and-Seek dataset card: https://huggingface.co/datasets/TUM-ICS/Hide-and-Seek

## Audit integrity

- Audit date: 2026-11-26.
- Method: official public dataset cards and project pages only.
- No full dataset download performed.
- No per-object counts independently computed.
- The 5 N per-sensor denoising threshold and 200 N scaling reference were located in the paper text; their use for our trajectory-level onset rule remains unvalidated.
- No model trained.
- No benchmark performance claimed.
