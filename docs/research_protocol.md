# Research Protocol

## 1. Research question

Can a model adapt to a previously unseen tactile object or contact structure from a small support set, using pressure-only tactile observations, and generalize to held-out examples?

The exact operational definition of "structure" must be grounded in the selected dataset's annotation schema. Do not assume that object identity, geometry, contact pattern, and action label are interchangeable.

## 2. Current research choices

- Problem family: few-shot adaptation.
- Adaptation target: structural variation in tactile interaction.
- Data boundary: public datasets.
- Input modality: pressure-only for the primary study.
- Candidate meta-learning algorithm: Reptile.
- Required comparator: ordinary model initialization followed by task-specific fine-tuning.

These are research directions, not evidence that a suitable dataset or successful method has already been established.

## 3. Dataset acceptance gates

A dataset can enter the primary experiment only after documenting:
1. Public availability, source URL, license, and access date.
2. Sensor modality and whether pressure signals can be isolated without video.
3. Sensor dimensions, sampling rate, calibration, missingness, and preprocessing requirements.
4. Object, contact, task, subject, session, and trial identifiers where available.
5. Number of independent groups and examples per group.
6. Whether support/query tasks can be formed without leakage.
7. Known limitations and risks of confounding.

If a required field is absent, mark it unavailable rather than infer it.

## 4. Task construction and leakage control

- Define task identity from documented labels only.
- Keep correlated trials, repeated recordings, and near-duplicate windows in the same split where possible.
- Separate meta-training tasks from meta-validation and meta-test tasks.
- Test on held-out objects/structures according to the stated research question.
- Fit normalization and feature transforms using training data only.
- Document every exclusion and split seed.

## 5. Methods

### Baseline A: pooled supervised model
Train on meta-training data and evaluate on held-out tasks without task-specific adaptation.

### Baseline B: fine-tuning
Use the same architecture and training data as the meta-learning method. Adapt from the common initialization using only the support set. Match inner-loop update budget and report sensitivity to the budget.

### Candidate: Reptile
Train an initialization across sampled training tasks. For each meta-iteration, adapt on a task's support data and move the initialization toward the adapted parameters. Evaluate adaptation on held-out tasks using support data only; report query performance separately.

Do not claim Reptile is superior unless the paired evaluation supports that conclusion.

## 6. Evaluation

Choose metrics only after confirming task type and label semantics. Report:
- per-task and aggregate performance;
- mean and dispersion across task seeds;
- support-set size and adaptation-step sensitivity;
- performance before and after adaptation;
- compute budget and runtime;
- confidence intervals or suitable uncertainty summaries.

Use paired task-level comparisons where possible. Avoid treating correlated windows as independent samples.

## 7. Reporting rules

Every result must be traceable to a committed configuration, code revision, dataset version, seed, and raw log. Report failed runs and deviations. Distinguish exploratory results from confirmatory experiments.

## 8. Decision gates

Do not begin the main meta-learning comparison until dataset audit, split design, and leakage tests pass. If the data cannot support defensible held-out structural tasks, revise the question or dataset choice before training.
