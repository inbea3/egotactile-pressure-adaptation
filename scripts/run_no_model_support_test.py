#!/usr/bin/env python3
"""Stage 2 no-model repeatability and participant-level Support-information test."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

REGIONS = ("thumb", "index", "middle", "ring", "little", "palm")
SHOTS = (1, 2, 5)


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        return np.full(len(vector), 1.0 / len(vector))
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    candidates = ordered - cumulative / np.arange(1, len(vector) + 1) > 0
    if not candidates.any():
        return np.full(len(vector), 1.0 / len(vector))
    rho = int(np.flatnonzero(candidates)[-1])
    theta = cumulative[rho] / float(rho + 1)
    return np.maximum(vector - theta, 0.0)


def extract_participant_vectors(stage1_report: dict, recording_policy: str = "all_detected_events") -> dict:
    if recording_policy not in {"all_detected_events", "exactly_five_events"}:
        raise ValueError(f"Unknown recording policy: {recording_policy}")
    objects = {}
    for obj in stage1_report["objects"]:
        participants = {}
        for episode in obj["episodes"]:
            if episode.get("excluded_from_signal_diagnostics", False):
                continue
            cycles = episode.get("grasp_cycle_candidates", {}).get("0.25", {}).get("cycles", [])
            if recording_policy == "exactly_five_events" and len(cycles) != 5:
                continue
            valid = [
                cycle for cycle in cycles
                if cycle.get("early_region_load_share") is not None
                and cycle.get("future_region_load_share") is not None
                and cycle.get("early_future_window_covered_without_next_cycle_overlap", False)
            ]
            if not valid or (recording_policy == "exactly_five_events" and len(valid) != 5):
                continue
            early = np.asarray([x["early_region_load_share"] for x in valid], dtype=np.float64)
            future = np.asarray([x["future_region_load_share"] for x in valid], dtype=np.float64)
            participant = episode["subject_id"]
            if participant not in participants:
                participants[participant] = {"early": [], "future": [], "valid_cycle_count": 0}
            participants[participant]["early"].append(early.mean(axis=0))
            participants[participant]["future"].append(future.mean(axis=0))
            participants[participant]["valid_cycle_count"] += len(valid)

        normalized = {}
        for participant, values in participants.items():
            early = np.mean(values["early"], axis=0)
            future = np.mean(values["future"], axis=0)
            normalized[participant] = {
                "early": early,
                "future": future,
                "delta": future - early,
                "valid_cycle_count": values["valid_cycle_count"],
            }
        if normalized:
            objects[obj["object_name"]] = normalized
    return objects


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, reps: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan"), float("nan")]
    samples = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[samples].mean(axis=1)
    return [float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run_no_model_support_test(
    stage1_report: dict,
    bootstrap_reps: int = 5000,
    seed: int = 20261126,
    recording_policy: str = "all_detected_events",
) -> dict:
    data = extract_participant_vectors(stage1_report, recording_policy)
    all_object_names = {x["object_name"] for x in stage1_report["objects"]}
    common_objects = sorted(name for name, people in data.items() if len(people) >= 6)
    excluded_objects = sorted(all_object_names - set(common_objects))
    if len(common_objects) < 3:
        raise ValueError("At least three objects with six valid participants are required")

    rng = np.random.default_rng(seed)

    # Participant-level within-object versus between-object pairwise structure.
    repeatability = []
    all_within = []
    all_between = []
    same_participant_between = []
    for name in common_objects:
        people = data[name]
        participant_names = sorted(people)
        within = [
            _mae(people[p]["future"], people[q]["future"])
            for i, p in enumerate(participant_names)
            for q in participant_names[i + 1:]
        ]
        between = []
        for other in common_objects:
            if other == name:
                continue
            for p, x in people.items():
                for q, y in data[other].items():
                    distance = _mae(x["future"], y["future"])
                    between.append(distance)
                    if p == q:
                        same_participant_between.append(distance)
        within_mean = float(np.mean(within)) if within else None
        between_mean = float(np.mean(between)) if between else None
        all_within.extend(within)
        all_between.extend(between)
        repeatability.append({
            "object_name": name,
            "participant_count": len(people),
            "within_object_participant_pairwise_mae": within_mean,
            "between_object_participant_pairwise_mae": between_mean,
            "within_less_than_between": bool(within_mean < between_mean) if within_mean is not None and between_mean is not None else None,
        })

    # Leave-one-object-out global transition and participant-disjoint Support correction.
    object_metrics = {str(k): [] for k in SHOTS}
    for heldout in common_objects:
        query_data = data[heldout]
        other_objects = [name for name in common_objects if name != heldout]
        global_delta = np.mean([
            np.mean([x["delta"] for x in data[name].values()], axis=0)
            for name in other_objects
        ], axis=0)
        global_future = np.mean([
            np.mean([x["future"] for x in data[name].values()], axis=0)
            for name in other_objects
        ], axis=0)

        for shot in SHOTS:
            query_rows = []
            people = sorted(query_data)
            for query_person in people:
                query = query_data[query_person]
                support_people = [p for p in people if p != query_person]
                subset_errors = []
                support_future_errors = []
                for subset in itertools.combinations(support_people, shot):
                    support_delta = np.mean([query_data[p]["delta"] for p in subset], axis=0)
                    support_prediction = project_to_simplex(query["early"] + support_delta)
                    support_future_prototype = project_to_simplex(np.mean([query_data[p]["future"] for p in subset], axis=0))
                    subset_errors.append(_mae(query["future"], support_prediction))
                    support_future_errors.append(_mae(query["future"], support_future_prototype))
                support_mae = float(np.mean(subset_errors))
                persistence_mae = _mae(query["future"], project_to_simplex(query["early"]))
                global_mae = _mae(query["future"], project_to_simplex(query["early"] + global_delta))
                global_future_mae = _mae(query["future"], project_to_simplex(global_future))
                query_rows.append({
                    "query_participant": query_person,
                    "support_subsets_evaluated": len(subset_errors),
                    "support_correction_mae": support_mae,
                    "support_future_prototype_mae": float(np.mean(support_future_errors)),
                    "global_transition_mae": global_mae,
                    "global_future_prototype_mae": global_future_mae,
                    "early_persistence_mae": persistence_mae,
                })
            object_metrics[str(shot)].append({
                "object_name": heldout,
                "participant_count": len(people),
                "support_query_combinations": sum(x["support_subsets_evaluated"] for x in query_rows),
                "support_correction_mae": float(np.mean([x["support_correction_mae"] for x in query_rows])),
                "support_future_prototype_mae": float(np.mean([x["support_future_prototype_mae"] for x in query_rows])),
                "global_transition_mae": float(np.mean([x["global_transition_mae"] for x in query_rows])),
                "global_future_prototype_mae": float(np.mean([x["global_future_prototype_mae"] for x in query_rows])),
                "early_persistence_mae": float(np.mean([x["early_persistence_mae"] for x in query_rows])),
                "query_details": query_rows,
            })

    summaries = {}
    for shot in SHOTS:
        rows = object_metrics[str(shot)]
        support = np.asarray([x["support_correction_mae"] for x in rows])
        global_ = np.asarray([x["global_transition_mae"] for x in rows])
        persistence = np.asarray([x["early_persistence_mae"] for x in rows])
        support_proto = np.asarray([x["support_future_prototype_mae"] for x in rows])
        global_proto = np.asarray([x["global_future_prototype_mae"] for x in rows])
        delta_global = global_ - support
        delta_persistence = persistence - support
        summaries[str(shot)] = {
            "object_count": len(rows),
            "support_correction_mae_mean_ci95": _bootstrap_mean(support, rng, bootstrap_reps),
            "support_future_prototype_mae_mean_ci95": _bootstrap_mean(support_proto, rng, bootstrap_reps),
            "global_transition_mae_mean_ci95": _bootstrap_mean(global_, rng, bootstrap_reps),
            "global_future_prototype_mae_mean_ci95": _bootstrap_mean(global_proto, rng, bootstrap_reps),
            "global_future_prototype_minus_support_future_prototype_improvement_mean_ci95": _bootstrap_mean(global_proto - support_proto, rng, bootstrap_reps),
            "early_persistence_mae_mean_ci95": _bootstrap_mean(persistence, rng, bootstrap_reps),
            "global_minus_support_improvement_mean_ci95": _bootstrap_mean(delta_global, rng, bootstrap_reps),
            "persistence_minus_support_improvement_mean_ci95": _bootstrap_mean(delta_persistence, rng, bootstrap_reps),
            "objects_support_beats_global": int(np.sum(delta_global > 0)),
            "objects_support_beats_persistence": int(np.sum(delta_persistence > 0)),
            "objects_support_future_prototype_beats_global_future_prototype": int(np.sum(global_proto > support_proto)),
        }

    within_object_means = np.asarray([x["within_object_participant_pairwise_mae"] for x in repeatability])
    between_object_means = np.asarray([x["between_object_participant_pairwise_mae"] for x in repeatability])
    repeatability_summary = {
        "within_object_pairwise_mae_mean_ci95_over_objects": _bootstrap_mean(within_object_means, rng, bootstrap_reps),
        "between_object_pairwise_mae_mean_ci95_over_objects": _bootstrap_mean(between_object_means, rng, bootstrap_reps),
        "same_participant_between_object_pairwise_mae_mean": float(np.mean(same_participant_between)) if same_participant_between else None,
        "within_pair_count": len(all_within),
        "between_pair_count": len(all_between),
        "same_participant_between_pair_count": len(same_participant_between),
    }

    return {
        "stage": "Stage 2 exploratory no-model Support information test",
        "dataset_revision": stage1_report.get("revision"),
        "recording_policy": recording_policy,
        "object_count_common_5shot_set": len(common_objects),
        "common_objects": common_objects,
        "objects_excluded_for_less_than_six_valid_participants": excluded_objects,
        "participant_level_recording_unit": True,
        "events_averaged_within_participant_before_inference": True,
        "support_query_participants_are_disjoint_within_object": True,
        "global_transition_is_leave_one_object_out": True,
        "target_regions": list(REGIONS),
        "repeatability": repeatability,
        "repeatability_summary": repeatability_summary,
        "repeatability_objects_within_less_than_between": sum(x["within_less_than_between"] is True for x in repeatability),
        "shot_summaries": summaries,
        "object_level_metrics": object_metrics,
        "bootstrap_reps": bootstrap_reps,
        "seed": seed,
        "model_training_performed": False,
        "interpretation_boundary": "Exploratory feasibility test only. Named objects are the inferential units; participant recordings are Support/Query units; repeated pressure events are nested and averaged within participant.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_stage2_no_model_support_test.json"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    results = {
        "all_detected_events": run_no_model_support_test(report, args.bootstrap_reps, args.seed, "all_detected_events"),
        "exactly_five_events": run_no_model_support_test(report, args.bootstrap_reps, args.seed, "exactly_five_events"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        policy: {
            "objects": result["object_count_common_5shot_set"],
            "repeatability_within_less_than_between": result["repeatability_objects_within_less_than_between"],
            "repeatability": result["repeatability_summary"],
            "shots": result["shot_summaries"],
        }
        for policy, result in results.items()
    } | {"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
