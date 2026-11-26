# EgoTactile Pressure Adaptation

Research workspace for studying few-shot adaptation to **new tactile objects and contact structures** using pressure-only tactile observations.

> Research principle: establish what the public data actually contains before selecting tasks, splits, metrics, or making performance claims. No results are fabricated.

## Research scope

- **Core problem:** adaptation to previously unseen objects / contact structures.
- **Adaptation target:** structural information in pressure-based tactile observations.
- **Data policy:** public datasets only; dataset suitability must be audited and documented.
- **Modality boundary:** pressure-only experiments. Do not silently introduce synchronized video or other modalities.
- **Methodology:** evaluate meta-learning where justified, with Reptile as a candidate rather than a presumed winner.
- **Baselines:** ordinary supervised training and fair within-task fine-tuning must be implemented under comparable data and compute budgets.

## Repository boundaries

This repository is independent of `inbea3/touch`. Do not modify that repository, its branches, files, or experiment outputs. Any reuse of prior code or findings must be explicitly reviewed, attributed, and revalidated here.

## Current status

- Repository initialization is complete.
- Primary dataset is fixed: HustleHard/EgoTactile-OXT, gloved-hand pressure-only subset.
- Hub revision pinned: `5b05f19e613758e76c330671f4c9f3c06af8af44`; 63 object ZIP archives and 3,459,799,746 total archive bytes verified by a successful metadata-only GitHub Actions run.
- Stage 0 full audit completed: all 63 object archives, 314,100 frames, and 767 episodes were inventoried using pressure/timestamp/metadata arrays only.
- 692 directly paired gloved-hand participant-object recordings were identified across 63 objects (62 objects have 11 recordings; Sponge has 10). Every object is count-feasible for 5-shot Support plus one independent participant Query. The synchronized master clock is valid; tactile source-clock anomalies are retained as warnings, not primary exclusions.
- All objects have a stable 162-layout to 137-unique-physical-index mapping and finite pressure arrays. Timestamp anomalies and exclusions are documented.
- Unit tests are passing. The official mapping note has been parsed and the 162-layout to 137-physical-sensor crosswalk passed on three objects. Six anatomical load regions are defined. Stage 1 now focuses on timestamp/pressure semantics, contact onset, and regional target construction.
- No model has been trained and no experimental performance result is claimed.

See [the pinned dataset revision manifest](docs/dataset_revision_manifest_20261126.md), [frozen experimental protocol](docs/frozen_experimental_protocol_v1.md), [Stage 0 sample audit](docs/stage0_sample_pressure_audit_20261126.md), and [Stage 0 full inventory](docs/stage0_full_inventory_20261126.md).

## Planned workflow

1. Complete EgoTactile-OXT archive and trajectory inventory.
2. Verify pressure sensor mapping, timestamps, episode boundaries, and signal quality.
3. Freeze contact onset, early/future windows, and structural target.
4. Run structure-repeatability and no-model Support-information tests.
5. Only after the feasibility gate passes, run object-disjoint MLP and Reptile comparisons.
6. Run sensitivity, uncertainty, and robustness analyses.

## Reproducibility

Every experiment should record its configuration, random seed, dataset version/checksum, code commit, environment, metrics, and logs. Never overwrite raw data or prior experiment outputs.

See [research protocol](docs/research_protocol.md), [data audit protocol](docs/data_audit_protocol.md), [candidate dataset audit](docs/dataset_candidate_audit_20261126.md), and [experiment tracking policy](docs/experiment_tracking.md).
