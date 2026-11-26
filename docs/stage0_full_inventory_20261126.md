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
| Eligible preliminary gloved-hand trajectories after conservative signal/time exclusions | **692** |
| Objects with at least 6 eligible trajectories | **63 / 63** |

The 692 direct gloved-hand episodes and 75 weakly paired bare-hand episodes reconcile exactly with the 767 episode count reported by the dataset card.

## 3. Shot feasibility by object

Use the synchronized master clock in `data.timestamps` for the pressure sequence. The source glove clock `data.tactile_timestamps` can repeat while the latest pressure readout is held at a master tick; its short durations and duplicate-time warnings are not grounds to exclude a master-clock-valid recording.

The full archive inventory contains 692 directly paired gloved-hand participant-object recordings: 62 objects have 11 recordings and Sponge has 10. All 63 named objects satisfy the minimum six-participant requirement for 5-shot Support plus one independent participant Query. This is a **data-count feasibility result**, not evidence that Support contains predictive information.

The inferential task remains new **named-object** adaptation. The seven broad categories are not the held-out unit and are not called unseen categories.

## 4. Exclusions and integrity exceptions

Three source-clock timing anomalies and one source-clock duplicate-pressure warning were found. They are not primary exclusions when the synchronized master-clock pressure sequence is valid:

| Object | Participant | Reason |
|---|---|---|
| GlassBowl | p001 | Tactile source clock spans only 1.084 s, while master clock spans 28.632 s; source-clock warning only |
| Orange | p001 | Tactile source clock spans 2.105 s, while master clock spans 28.247 s; source-clock warning only |
| Toothpaste | p002 | Tactile source clock is constant across 394 master frames; source-clock warning only |
| BellPepper | p002 | One duplicate tactile source timestamp carries a different pressure vector; source-clock warning only |

A second duplicate-time pressure conflict was found in SamyangSpicyChickenCupNoodles, participant p012, a weakly paired bare-hand record already excluded from the primary subset.

The three short-duration trajectories are **not** rescued using frame timestamps. That would assume synchronization semantics not yet established for the pressure stream. They remain excluded until a separately justified data-repair analysis is approved.

## 5. Timestamp semantics

Across the three initial sample objects, repeated adjacent tactile timestamps were accompanied by exactly repeated pressure vectors. The full inventory found two object-level duplicate-time pressure conflicts; focused episode inspection located:
- one in a primary gloved-hand trajectory (BellPepper/p002);
- one in an already-excluded weakly paired bare-hand trajectory (SamyangSpicyChickenCupNoodles/p012).

The full inventory found no negative tactile timestamp intervals within the individual episode slices. Some concatenated object arrays contain a negative interval at an episode boundary; this is why all temporal checks are performed within episode boundaries, never across the concatenated object array.

The corrected frozen rule is:
- use `data.timestamps` as the synchronized master time axis for the aligned pressure sequence;
- keep every pressure row at its distinct master timestamp, including repeated pressure values held between glove updates;
- retain `data.tactile_timestamps` only as an auxiliary source-clock diagnostic;
- process episodes independently because source clocks may reset at episode boundaries;
- do not collapse master-clock samples based on repeated source tactile timestamps.

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

**GO: proceed to Stage 1 pressure-event and target construction.**

The dataset has enough independent full trajectories for 1/2/5-shot experiments across all 63 named objects after conservative preliminary exclusions.

**Not yet approved:**
- final contact-onset detector;
- final 1 s / 2 s window eligibility;
- regional load-share target;
- structure-repeatability conclusion;
- any neural model training or Reptile experiment.

Stage 1 has validated the official tactile mapping and a 12-object pressure-only signal sample. The next step is to run the candidate pressure-event detector and target construction across all 63 objects, then freeze the event/window/target rules without reference to model performance.
