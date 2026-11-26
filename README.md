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

The project is at initialization. No dataset has yet been certified as suitable, and no experimental result is claimed.

## Planned workflow

1. Audit candidate public datasets and licensing/access conditions.
2. Verify modality, sensor layout, object/structure labels, subject/session identifiers, and sample counts.
3. Define leakage-resistant task construction and train/validation/test splits.
4. Implement data loaders and reproducible preprocessing.
5. Establish non-meta-learning baselines.
6. Implement and test Reptile; compare fairly against fine-tuning.
7. Run ablations, uncertainty analysis, and robustness checks.
8. Preserve raw logs, configurations, code revision, and derived results.

## Reproducibility

Every experiment should record its configuration, random seed, dataset version/checksum, code commit, environment, metrics, and logs. Never overwrite raw data or prior experiment outputs.

See [research protocol](docs/research_protocol.md), [data audit protocol](docs/data_audit_protocol.md), and [experiment tracking policy](docs/experiment_tracking.md).
