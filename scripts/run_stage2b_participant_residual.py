#!/usr/bin/env python3
"""Stage 2B: participant-controlled, object-held-out Support diagnostics.

Consumes the frozen Stage 1 derived-feature report. No raw pressure or video is
read, no model is fitted, and no Query outcome is used to select a method.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .run_no_model_support_test import extract_participant_vectors, project_to_simplex
except ImportError:
    from run_no_model_support_test import extract_participant_vectors, project_to_simplex

SHOTS = (1, 2, 5)
POLICIES = ("all_detected_events", "exactly_five_events")
FIXED_SUPPORT_WEIGHT = 0.25  # Exploratory value reported in Stage 2; never selected on Query.


def mae(left, right):
    return float(np.mean(np.abs(np.asarray(left) - np.asarray(right))))


def object_mean(people, key, exclude_subject=None):
    values = [row[key] for subject, row in people.items() if subject != exclude_subject]
    return np.mean(values, axis=0) if values else None


def global_mean(data, names, key, exclude_subject=None):
    means = [object_mean(data[name], key, exclude_subject) for name in names]
    available = [value for value in means if value is not None]
    if not available:
        raise ValueError("No training recordings remain after participant exclusion")
    return np.mean(available, axis=0)


def bootstrap_mean(values, rng, repetitions):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite object metric")
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
    }


def residual_diagnostics(data, names):
    """Compare future-minus-early structure across objects and participants."""
    rows = []
    for name in names:
        people = data[name]
        other_names = [other for other in names if other != name]
        within = [
            mae(people[a]["delta"], people[b]["delta"])
            for a, b in itertools.combinations(sorted(people), 2)
        ]
        between_same_subject = [
            mae(row["delta"], data[other][subject]["delta"])
            for subject, row in people.items()
            for other in other_names if subject in data[other]
        ]
        query_predictions = []
        for subject, query in people.items():
            object_template = object_mean(people, "delta", exclude_subject=subject)
            actor_values = [data[other][subject]["delta"] for other in other_names if subject in data[other]]
            actor_template = np.mean(actor_values, axis=0) if actor_values else None
            global_template = global_mean(data, other_names, "delta", exclude_subject=subject)
            query_predictions.append({
                "global_mae": mae(query["delta"], global_template),
                "object_mae": mae(query["delta"], object_template),
                "actor_mae": mae(query["delta"], actor_template) if actor_template is not None else None,
            })
        rows.append({
            "object_name": name,
            "within_object_different_subject_residual_mae": float(np.mean(within)),
            "between_object_same_subject_residual_mae": float(np.mean(between_same_subject)) if between_same_subject else None,
            "global_residual_template_mae": float(np.mean([x["global_mae"] for x in query_predictions])),
            "object_residual_template_mae": float(np.mean([x["object_mae"] for x in query_predictions])),
            "actor_residual_template_mae": float(np.mean([x["actor_mae"] for x in query_predictions if x["actor_mae"] is not None])) if any(x["actor_mae"] is not None for x in query_predictions) else None,
        })
    return rows


def evaluate_policy(stage1_report, policy, repetitions=5000, seed=20261126):
    data = extract_participant_vectors(stage1_report, policy)
    names = sorted(name for name, people in data.items() if len(people) >= 6)
    if len(names) < 3:
        raise ValueError("At least three objects with six participants are needed")
    rng = np.random.default_rng(seed)
    residual = residual_diagnostics(data, names)
    object_rows = {str(shot): [] for shot in SHOTS}

    for heldout in names:
        people = data[heldout]
        training_names = [name for name in names if name != heldout]
        global_known = global_mean(data, training_names, "delta")
        for shot in SHOTS:
            query_rows = []
            for subject, query in sorted(people.items()):
                support_subjects = [other for other in sorted(people) if other != subject]
                global_blocked = global_mean(data, training_names, "delta", exclude_subject=subject)
                actor_values = [data[name][subject]["delta"] for name in training_names if subject in data[name]]
                actor_global = np.mean(actor_values, axis=0) if actor_values else None
                global_future_known = global_mean(data, training_names, "future")
                candidate = {
                    "query_subject": subject,
                    "support_combinations": 0,
                    "persistence": mae(query["future"], project_to_simplex(query["early"])),
                    "global_known_subject": mae(query["future"], project_to_simplex(query["early"] + global_known)),
                    "global_query_subject_blocked": mae(query["future"], project_to_simplex(query["early"] + global_blocked)),
                    "global_future_prototype": mae(query["future"], project_to_simplex(global_future_known)),
                    "actor_global": mae(query["future"], project_to_simplex(query["early"] + actor_global)) if actor_global is not None else None,
                }
                support_errors, blend_errors, prototype_errors = [], [], []
                for subset in itertools.combinations(support_subjects, shot):
                    support_delta = np.mean([people[p]["delta"] for p in subset], axis=0)
                    support_future = np.mean([people[p]["future"] for p in subset], axis=0)
                    support_errors.append(mae(query["future"], project_to_simplex(query["early"] + support_delta)))
                    blend = (1 - FIXED_SUPPORT_WEIGHT) * global_blocked + FIXED_SUPPORT_WEIGHT * support_delta
                    blend_errors.append(mae(query["future"], project_to_simplex(query["early"] + blend)))
                    prototype_errors.append(mae(query["future"], project_to_simplex(support_future)))
                candidate["support_combinations"] = len(support_errors)
                candidate["support_delta"] = float(np.mean(support_errors))
                candidate["fixed_quarter_shrinkage_blocked"] = float(np.mean(blend_errors))
                candidate["support_future_prototype"] = float(np.mean(prototype_errors))
                query_rows.append(candidate)

            methods = ["persistence", "global_known_subject", "global_query_subject_blocked",
                       "actor_global", "support_delta", "fixed_quarter_shrinkage_blocked",
                       "support_future_prototype", "global_future_prototype"]
            aggregate = {
                method: float(np.mean([row[method] for row in query_rows if row[method] is not None]))
                if any(row[method] is not None for row in query_rows) else None
                for method in methods
            }
            object_rows[str(shot)].append({
                "object_name": heldout,
                "query_participants": len(query_rows),
                "support_query_combinations": sum(row["support_combinations"] for row in query_rows),
                "metrics": aggregate,
            })

    summaries = {}
    for shot in SHOTS:
        rows = object_rows[str(shot)]
        methods = rows[0]["metrics"].keys()
        summary = {
            "object_count": len(rows),
            "methods": {
                method: bootstrap_mean([row["metrics"][method] for row in rows], rng, repetitions)
                for method in methods if all(row["metrics"][method] is not None for row in rows)
            },
            "paired_differences_positive_favors_second": {},
        }
        for baseline, adapted in [
            ("global_query_subject_blocked", "support_delta"),
            ("global_query_subject_blocked", "fixed_quarter_shrinkage_blocked"),
            ("global_known_subject", "support_delta"),
            ("global_known_subject", "fixed_quarter_shrinkage_blocked"),
            ("persistence", "fixed_quarter_shrinkage_blocked"),
            ("global_future_prototype", "support_future_prototype"),
        ]:
            differences = [row["metrics"][baseline] - row["metrics"][adapted] for row in rows]
            summary["paired_differences_positive_favors_second"][f"{baseline}_minus_{adapted}"] = {
                **bootstrap_mean(differences, rng, repetitions),
                "objects_improved": int(np.sum(np.asarray(differences) > 0)),
            }
        summaries[str(shot)] = summary

    residual_summary = {}
    for key in ["within_object_different_subject_residual_mae", "between_object_same_subject_residual_mae",
                "global_residual_template_mae", "object_residual_template_mae", "actor_residual_template_mae"]:
        values = [row[key] for row in residual if row[key] is not None]
        residual_summary[key] = bootstrap_mean(values, rng, repetitions) if values else None
    differences = [row["global_residual_template_mae"] - row["object_residual_template_mae"] for row in residual]
    residual_summary["global_minus_object_residual_template"] = {
        **bootstrap_mean(differences, rng, repetitions),
        "objects_improved": int(np.sum(np.asarray(differences) > 0)),
    }

    return {
        "stage": "Stage 2B exploratory participant-controlled Support and residual audit",
        "dataset_revision": stage1_report.get("revision"),
        "recording_policy": policy,
        "shot_unit": "one participant-object recording; its pressure events remain nested",
        "query_subject_excluded_from_blocked_global": True,
        "global_known_subject_is_historical_comparator": True,
        "support_weight": FIXED_SUPPORT_WEIGHT,
        "support_weight_status": "previously inspected exploratory value, not independently validated",
        "object_count": len(names),
        "residual_summary": residual_summary,
        "residual_by_object": residual,
        "shot_summaries": summaries,
        "object_metrics": object_rows,
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "interpretation": "All 63 objects have been used in exploratory study design. No result is an untouched confirmatory test.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/stage2b_participant_residual.json"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    stage1_report = json.loads(args.input.read_text(encoding="utf-8"))
    if stage1_report.get("revision") != "5b05f19e613758e76c330671f4c9f3c06af8af44":
        raise ValueError("Stage 1 input is not the frozen EgoTactile-OXT revision")
    results = {policy: evaluate_policy(stage1_report, policy, args.bootstrap_reps, args.seed) for policy in POLICIES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({policy: {
        "objects": result["object_count"],
        "residual": result["residual_summary"],
        "shots": result["shot_summaries"],
    } for policy, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()

