# Stage 0 Full EgoTactile Pressure-Only Inventory — 2026-11-26

## Provenance

- Dataset: `HustleHard/EgoTactile-OXT`
- Pinned revision: `5b05f19e613758e76c330671f4c9f3c06af8af44`
- License: CC BY-NC 4.0
- Full inventory workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45801252471
- Full inventory artifact: `15107476138`
- Artifact SHA-256: `5888efff04773c69d2b71e38f59172780a250a8fece4f78d421473c257fde21e`
- Focused duplicate-timestamp audit workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45801856998
- Focused short-duration audit workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45802196704
- Raw object archives downloaded by the full inventory: 63 (3,459,799,746 bytes)
- RGB/video arrays decoded: none
- Raw pressure values exported to artifact/repository: none

## 1. Recording-unit clarification

Each OXT episode is a participant-object recording containing approximately five grasp repetitions, not a single grasp. The paper's acquisition protocol describes five repetitions per object per participant, each about six seconds. Therefore, 1/2/5-shot feasibility below means 1/2/5 distinct participant-object recordings (participants) for Support plus one different participant recording for Query. The five repetitions inside a recording are never split across Support/Query and are not treated as independent observations.

## 2. Dataset-level inventory

| Quantity | Observed |
|---|---:|
| Named object archives | 63 |
| Total frames | 314,100 |
| Complete episodes / source trajectories | 767 |
| Direct gloved-hand episodes | 692 |
| Weakly paired bare-hand episodes | 75 |
| Eligible preliminary gloved-hand trajectories after conservative signal/time exclusions | **688** |
| Objects with at least 6 eligible trajectories | **63 / 63** |

The 692 direct gloved-hand episodes and 75 weakly paired bare-hand episodes reconcile exactly with the 767 episode count reported by the dataset card.

## 3. Shot feasibility by object

After applying the follow-up episode-level timestamp-conflict check, eligible trajectory counts are:
- 58 objects with 11 eligible trajectories;
- 5 objects with 10 eligible trajectories.

Therefore all 63 named objects satisfy the minimum six-trajectory requirement for 5-shot Support plus one independent Query. This is a **data-count feasibility result**, not evidence that Support contains predictive information.

The inferential task remains new **named-object** adaptation. The seven broad categories are not the held-out unit and are not called unseen categories.

## 4. Exclusions and integrity exceptions

Four direct gloved-hand trajectories are conservatively excluded from the preliminary 1-second + 2-second window candidate pool:

| Object | Participant | Reason |
|---|---|---|
| GlassBowl | p001 | Tactile timestamp elapsed duration 1.084 s despite 28.632 s frame timestamp duration; only 9 unique tactile timestamps |
| Orange | p001 | Tactile timestamp elapsed duration 2.105 s despite 28.247 s frame timestamp duration; only 10 unique tactile timestamps |
| Toothpaste | p002 | Tactile timestamps are constant across 394 frames (0 s elapsed); only 1 unique tactile timestamp |
| BellPepper | p002 | One duplicate tactile timestamp carries a pressure vector different from the preceding row; timestamp/pressure conflict |

A second duplicate-time pressure conflict was found in SamyangSpicyChickenCupNoodles, participant p012, but that episode is a weakly paired bare-hand record and is already excluded from the primary subset.

The three short-duration trajectories are **not** rescued using frame timestamps. That would assume synchronization semantics not yet established for the pressure stream. They remain excluded until a separately justified data-repair analysis is approved.

## 5. Timestamp semantics

Across the three initial sample objects, repeated adjacent tactile timestamps were accompanied by exactly repeated pressure vectors. The full inventory found two object-level duplicate-time pressure conflicts; focused episode inspection located:
- one in a primary gloved-hand trajectory (BellPepper/p002);
- one in an already-excluded weakly paired bare-hand trajectory (SamyangSpicyChickenCupNoodles/p012).

The full inventory found no negative tactile timestamp intervals within the individual episode slices. Some concatenated object arrays contain a negative interval at an episode boundary; this is why all temporal checks are performed within episode boundaries, never across the concatenated object array.

The frozen rule is:
- collapse adjacent equal tactile timestamps only when pressure vectors are exactly equal;
- flag and exclude a trajectory if equal timestamps carry different pressure vectors;
- process episodes independently;
- use unique tactile timestamps for time-weighted windows;
- do not sort timestamps globally or substitute frame timestamps without validation.

The three short-duration anomalies demonstrate that the two timestamp fields cannot be assumed interchangeable.

## 6. Sensor mapping and pressure integrity

All 63 object archives were inspected:
- `right_tactile_data_pressure` has 162 layout positions;
- `right_tactile_raw_pressure_256` has 256 source values;
- `right_tactile_raw_index_pressure` has 162 indices;
- every object has exactly 137 unique physical raw sensor indices in the layout;
- the raw-index mapping is constant across all frames within every object archive;
- all 162-layout pressure values in all inspected archives are finite.

The per-object raw-index multiplicities and exact anatomical/region semantics must still be reconciled with `supplementary_tactile_note.json` before regional target construction. A 162-cell display is not 162 independent physical sensors.

## 7. Stage 0 decision

**GO: proceed to Stage 1 signal semantics and target construction.**

The dataset has enough independent full trajectories for 1/2/5-shot experiments across all 63 named objects after conservative preliminary exclusions.

**Not yet approved:**
- final contact-onset detector;
- final 1 s / 2 s window eligibility;
- regional load-share target;
- structure-repeatability conclusion;
- any neural model training or Reptile experiment.

Stage 1 must first retrieve and validate the official tactile mapping note, inspect timestamp/pressure behavior and baseline/noise distributions, define hand regions using documented mapping only, and freeze the onset/window/target rules without reference to model performance.
