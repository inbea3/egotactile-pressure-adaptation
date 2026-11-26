# Experiment Tracking and Reproducibility

Each run must have a unique run ID and preserve:

- Git commit SHA and working-tree status;
- dataset name, version, license/source, and checksum or manifest;
- preprocessing code revision and configuration;
- task construction and split manifest;
- model architecture and parameter count;
- optimizer, learning rates, batch sizes, update steps, and stopping rule;
- random seeds and software/hardware environment;
- support/query sizes and task counts;
- metrics before and after adaptation;
- per-task predictions/metrics where appropriate;
- stdout/stderr logs, runtime, and exit status.

## Directory convention

```text
results/
  <run_id>/
    config.yaml
    metadata.json
    metrics.csv
    per_task_metrics.csv
    stdout.log
    stderr.log
    summary.md
```

Never overwrite a completed run. Corrections should create a new run and explain the relationship to the earlier run.

A result is not considered validated merely because a script exits successfully. Check data counts, split integrity, finite losses, metric calculations, and reproducibility across seeds.
