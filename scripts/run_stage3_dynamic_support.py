#!/usr/bin/env python3
"""Pressure-only feasibility test for a causal-onset dynamic contact target.

Reads the frozen Stage 1 report's half-second regional load bins. It does not
read video or fit a neural model. See docs/stage3_dynamic_target_protocol.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path

import numpy as np

try:
    from .run_no_model_support_test import project_to_simplex
    from .run_stage2b_participant_residual import bootstrap_mean, global_mean, mae, object_mean
except ImportError:
    from run_no_model_support_test import project_to_simplex
    from run_stage2b_participant_residual import bootstrap_mean, global_mean, mae, object_mean


REGIONS = ("thumb", "index", "middle", "ring", "little", "palm")
ONSET_RULES = ("two_sensors_gt_5N_3samples", "any_sensor_gt_5N_3samples")
REVISION = "5b05f19e613758e76c330671f4c9f3c06af8af44"
SHOTS = (1, 2, 5)
CONTACT_LOAD_N = 10.0
BIN_SECONDS = 0.5
SUPPORT_WEIGHT = 0.25


def contact_row(bin_record: dict) -> np.ndarray | None:
    values = bin_record.get("region_load_mean_N", {})
    try:
        loads = np.asarray([values[name] for name in REGIONS], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(loads).all() or (loads < 0).any():
        return None
    total = float(loads.sum())
    result = np.zeros(7, dtype=np.float64)
    if total < CONTACT_LOAD_N:
        result[6] = 1.0
    else:
        result[:6] = loads / total
    return result


def recording_vector(episode: dict, onset_rule: str) -> tuple[dict | None, str | None]:
    if episode.get("excluded_from_signal_diagnostics", False):
        return None, "pressure_or_timestamp_qc"
    onset = episode.get("candidate_onsets", {}).get(onset_rule, {}).get("first_onset")
    if onset is None:
        return None, "no_causal_onset"
    onset_seconds = onset.get("onset_seconds_from_episode_start")
    if onset_seconds is None or not math.isfinite(float(onset_seconds)):
        return None, "invalid_onset_time"
    cutoff_bin = int(math.ceil((float(onset_seconds) + 1.0) / BIN_SECONDS - 1e-10))
    required = tuple(range(cutoff_bin - 2, cutoff_bin + 4))
    if required[0] < 0:
        return None, "too_early_for_two_observation_bins"

    trace = {}
    for row in episode.get("time_binned_pressure_trace_0p5s", []):
        start = row.get("time_bin_start_seconds")
        if start is None or not math.isfinite(float(start)):
            continue
        index = round(float(start) / BIN_SECONDS)
        if abs(float(start) - index * BIN_SECONDS) <= 1e-6:
            trace[index] = row
    if any(index not in trace for index in required):
        return None, "missing_required_time_bin"
    encoded = [contact_row(trace[index]) for index in required]
    if any(row is None for row in encoded):
        return None, "invalid_region_load_bin"
    early = np.stack(encoded[:2])
    future = np.stack(encoded[2:])
    return {
        "early": early,
        "future": future,
        "delta": future - early[-1][None, :],
        "cutoff_seconds_from_recording_start": cutoff_bin * BIN_SECONDS,
    }, None


def extract_vectors(stage1_report: dict, onset_rule: str) -> tuple[dict, dict]:
    if onset_rule not in ONSET_RULES:
        raise ValueError(f"Unknown onset rule: {onset_rule}")
    objects = {}
    reasons = Counter()
    eligible_recordings = 0
    for obj in stage1_report["objects"]:
        participants = {}
        for episode in obj["episodes"]:
            vector, reason = recording_vector(episode, onset_rule)
            if reason is not None:
                reasons[reason] += 1
                continue
            eligible_recordings += 1
            subject = episode["subject_id"]
            participants.setdefault(subject, []).append(vector)
        objects[obj["object_name"]] = {
            subject: {
                "early": np.mean([item["early"] for item in rows], axis=0),
                "future": np.mean([item["future"] for item in rows], axis=0),
            }
            for subject, rows in participants.items()
        }
        for row in objects[obj["object_name"]].values():
            row["delta"] = row["future"] - row["early"][-1][None, :]
    eligible_names = sorted(name for name, people in objects.items() if len(people) >= 6)
    return {name: objects[name] for name in eligible_names}, {
        "episodes_inspected": sum(len(obj["episodes"]) for obj in stage1_report["objects"]),
        "eligible_recordings": eligible_recordings,
        "eligible_objects": len(eligible_names),
        "excluded_objects": sorted(set(objects) - set(eligible_names)),
        "excluded_recordings_by_reason": dict(reasons),
        "participant_counts_by_object": {name: len(people) for name, people in objects.items()},
    }


def project_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4, 7):
        raise ValueError(f"Expected four seven-channel future bins, got {array.shape}")
    return np.stack([project_to_simplex(row) for row in array])


def residual_audit(data: dict, names: list[str], rng, repetitions: int) -> tuple[list, dict]:
    rows = []
    for heldout in names:
        people = data[heldout]
        other_names = [name for name in names if name != heldout]
        query_rows = []
        actor_templates = {
            subject: np.mean([data[name][subject]["delta"] for name in other_names if subject in data[name]], axis=0)
            for subject in people
            if any(subject in data[name] for name in other_names)
        }
        for subject, query in people.items():
            object_template = object_mean(people, "delta", exclude_subject=subject)
            blocked_global = global_mean(data, other_names, "delta", exclude_subject=subject)
            actor_template = actor_templates.get(subject)
            centered_support = [
                people[other]["delta"] - actor_templates[other]
                for other in people if other != subject and other in actor_templates
            ]
            centered_query = query["delta"] - actor_template if actor_template is not None else None
            query_rows.append({
                "object": mae(query["delta"], object_template),
                "blocked_global": mae(query["delta"], blocked_global),
                "actor": mae(query["delta"], actor_template) if actor_template is not None else None,
                "actor_centered_zero": mae(centered_query, np.zeros_like(centered_query))
                if centered_query is not None else None,
                "actor_centered_object": mae(centered_query, np.mean(centered_support, axis=0))
                if centered_query is not None and centered_support else None,
            })
        rows.append({
            "object_name": heldout,
            "object_template_mae": float(np.mean([row["object"] for row in query_rows])),
            "blocked_global_template_mae": float(np.mean([row["blocked_global"] for row in query_rows])),
            "actor_template_mae": float(np.mean([row["actor"] for row in query_rows if row["actor"] is not None]))
            if any(row["actor"] is not None for row in query_rows) else None,
            "actor_centered_zero_mae": float(np.mean([row["actor_centered_zero"] for row in query_rows if row["actor_centered_zero"] is not None]))
            if any(row["actor_centered_zero"] is not None for row in query_rows) else None,
            "actor_centered_object_mae": float(np.mean([row["actor_centered_object"] for row in query_rows if row["actor_centered_object"] is not None]))
            if any(row["actor_centered_object"] is not None for row in query_rows) else None,
        })
    differences = [row["blocked_global_template_mae"] - row["object_template_mae"] for row in rows]
    summary = {
        "object_template_mae": bootstrap_mean([row["object_template_mae"] for row in rows], rng, repetitions),
        "blocked_global_template_mae": bootstrap_mean([row["blocked_global_template_mae"] for row in rows], rng, repetitions),
        "blocked_global_minus_object_template": {
            **bootstrap_mean(differences, rng, repetitions),
            "objects_improved": int(np.sum(np.asarray(differences) > 0)),
        },
    }
    actor = [row["actor_template_mae"] for row in rows if row["actor_template_mae"] is not None]
    summary["actor_template_mae"] = bootstrap_mean(actor, rng, repetitions) if actor else None
    centered = [row for row in rows if row["actor_centered_zero_mae"] is not None and row["actor_centered_object_mae"] is not None]
    if centered:
        differences = [row["actor_centered_zero_mae"] - row["actor_centered_object_mae"] for row in centered]
        summary["actor_centered_zero_minus_object_template"] = {
            **bootstrap_mean(differences, rng, repetitions),
            "objects_improved": int(np.sum(np.asarray(differences) > 0)),
            "objects_evaluated": len(centered),
        }
    return rows, summary


def evaluate_rule(stage1_report: dict, onset_rule: str, repetitions=5000, seed=20261127) -> dict:
    data, inventory = extract_vectors(stage1_report, onset_rule)
    names = sorted(data)
    if len(names) < 3:
        raise ValueError("At least three objects with six eligible participants are required")
    rng = np.random.default_rng(seed)
    residual_rows, residual_summary = residual_audit(data, names, rng, repetitions)
    object_metrics = {str(shot): [] for shot in SHOTS}

    for heldout in names:
        people = data[heldout]
        other_names = [name for name in names if name != heldout]
        known_global = global_mean(data, other_names, "delta")
        for shot in SHOTS:
            query_rows = []
            for subject, query in sorted(people.items()):
                baseline = np.repeat(query["early"][-1][None, :], 4, axis=0)
                blocked_global = global_mean(data, other_names, "delta", exclude_subject=subject)
                actor = [data[name][subject]["delta"] for name in other_names if subject in data[name]]
                actor_global = np.mean(actor, axis=0) if actor else None
                support_people = [name for name in sorted(people) if name != subject]
                support_errors, shrinkage_errors = [], []
                actor_centered_errors, actor_centered_shrinkage_errors = [], []
                support_actor_baselines = {
                    name: np.mean([data[other][name]["delta"] for other in other_names if name in data[other]], axis=0)
                    for name in support_people
                    if any(name in data[other] for other in other_names)
                }
                for subset in itertools.combinations(support_people, shot):
                    support_delta = np.mean([people[name]["delta"] for name in subset], axis=0)
                    support_errors.append(mae(query["future"], project_rows(baseline + support_delta)))
                    correction = (1 - SUPPORT_WEIGHT) * blocked_global + SUPPORT_WEIGHT * support_delta
                    shrinkage_errors.append(mae(query["future"], project_rows(baseline + correction)))
                    if actor_global is not None and all(name in support_actor_baselines for name in subset):
                        object_effect = np.mean([
                            people[name]["delta"] - support_actor_baselines[name] for name in subset
                        ], axis=0)
                        actor_centered_errors.append(mae(query["future"], project_rows(baseline + actor_global + object_effect)))
                        actor_centered_shrinkage_errors.append(mae(query["future"], project_rows(baseline + actor_global + SUPPORT_WEIGHT * object_effect)))
                query_rows.append({
                    "support_combinations": len(support_errors),
                    "persistence": mae(query["future"], baseline),
                    "global_known_subject": mae(query["future"], project_rows(baseline + known_global)),
                    "global_query_subject_blocked": mae(query["future"], project_rows(baseline + blocked_global)),
                    "actor_global": mae(query["future"], project_rows(baseline + actor_global)) if actor_global is not None else None,
                    "support_delta": float(np.mean(support_errors)),
                    "fixed_quarter_shrinkage_blocked": float(np.mean(shrinkage_errors)),
                    "actor_centered_support": float(np.mean(actor_centered_errors)) if actor_centered_errors else None,
                    "actor_centered_quarter_shrinkage": float(np.mean(actor_centered_shrinkage_errors))
                    if actor_centered_shrinkage_errors else None,
                })
            methods = ("persistence", "global_known_subject", "global_query_subject_blocked",
                       "actor_global", "support_delta", "fixed_quarter_shrinkage_blocked",
                       "actor_centered_support", "actor_centered_quarter_shrinkage")
            object_metrics[str(shot)].append({
                "object_name": heldout,
                "query_participants": len(query_rows),
                "support_query_combinations": sum(row["support_combinations"] for row in query_rows),
                "metrics": {
                    method: float(np.mean([row[method] for row in query_rows if row[method] is not None]))
                    if any(row[method] is not None for row in query_rows) else None
                    for method in methods
                },
            })

    shot_summaries = {}
    for shot in SHOTS:
        rows = object_metrics[str(shot)]
        methods = rows[0]["metrics"]
        paired = {}
        for adapted in ("support_delta", "fixed_quarter_shrinkage_blocked"):
            differences = [row["metrics"]["global_query_subject_blocked"] - row["metrics"][adapted] for row in rows]
            paired[f"blocked_global_minus_{adapted}"] = {
                **bootstrap_mean(differences, rng, repetitions),
                "objects_improved": int(np.sum(np.asarray(differences) > 0)),
            }
        if all(row["metrics"]["actor_global"] is not None and row["metrics"]["actor_centered_quarter_shrinkage"] is not None for row in rows):
            differences = [row["metrics"]["actor_global"] - row["metrics"]["actor_centered_quarter_shrinkage"] for row in rows]
            paired["actor_global_minus_actor_centered_quarter_shrinkage"] = {
                **bootstrap_mean(differences, rng, repetitions),
                "objects_improved": int(np.sum(np.asarray(differences) > 0)),
            }
        shot_summaries[str(shot)] = {
            "object_count": len(rows),
            "methods": {
                method: bootstrap_mean([row["metrics"][method] for row in rows], rng, repetitions)
                for method in methods if all(row["metrics"][method] is not None for row in rows)
            },
            "paired_differences_positive_favors_adaptation": paired,
        }
    return {
        "stage": "Stage 3 exploratory causal-onset binned contact trajectory feasibility",
        "dataset_revision": stage1_report["revision"],
        "onset_rule": onset_rule,
        "contact_load_threshold_N": CONTACT_LOAD_N,
        "bin_seconds": BIN_SECONDS,
        "future_shape": [4, 7],
        "support_weight": SUPPORT_WEIGHT,
        "shot_unit": "participant-object recording; only first causal onset is used",
        "inventory": inventory,
        "residual_summary": residual_summary,
        "residual_by_object": residual_rows,
        "shot_summaries": shot_summaries,
        "object_metrics": object_metrics,
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "status": "exploratory; prior analyses used all objects",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("revision") != REVISION or source.get("object_count") != 63:
        raise ValueError("Stage 1 source must be the frozen full 63-object report")
    results = {
        rule: evaluate_rule(source, rule, args.bootstrap_reps, args.seed)
        for rule in ONSET_RULES
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({rule: {
        "inventory": report["inventory"],
        "residual": report["residual_summary"],
        "shots": report["shot_summaries"],
    } for rule, report in results.items()}, indent=2))


if __name__ == "__main__":
    main()

