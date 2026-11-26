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
                c for c in cycles
                if c.get("early_region_load_share") is not None
                and c.get("future_region_load_share") is not None
                and c.get("early_future_window_covered_without_next_cycle_overlap", False)
            ]
            if not valid or (recording_policy == "exactly_five_events" and len(valid) != 5):
                continue
            early = np.asarray([c["early_region_load_share"] for c in valid], dtype=np.float64)
            future = np.asarray([c["future_region_load_share"] for c in valid], dtype=np.float64)
            participant = episode["subject_id"]
            if participant not in participants:
                participants[participant] = {"early": [], "future": [], "valid_cycle_count": 0}
            participants[participant]["early"].append(early.mean(axis=0))
            participants[participant]["future"].append(future.mean(axis=0))
            participants[participant]["valid_cycle_count"] += len(valid)

        # If a participant has multiple episodes for one object, average at participant level.
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


def run_no_model_support_test(stage1_report: dict, bootstrap_reps: int = 5000, seed: int = 20261126, recording_policy: str = "all_detected_events") -> dict:
    data = extract_participant_vectors(stage1_report, recording_policy)
    common_objects = sorted(name for name, people in data.items() if len(people) >= 6)
    excluded_objects = sorted(set(data) - set(common_objects))
    if len(common_objects) < 3:
        raise ValueError("At least three objects with six valid participants are required for this exploratory test")

    # Participant-level within-object versus between-object pairwise structure.
    repeatability = []
    all_within = []
    all_between = []
    same_participant_between = []
    for name in common_objects:
        people = data[name]
        vectors = {p: x["future"] for p, x in people.items()}
        within = [
            _mae(vectors[p], vectors[q])
            for i, p in enumerate(sorted(vectors))
            for q in sorted(vectors)[i + 1:]
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

