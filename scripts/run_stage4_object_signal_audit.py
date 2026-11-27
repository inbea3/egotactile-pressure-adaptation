#!/usr/bin/env python3
"""Exploratory object-versus-participant pressure target audit on Stage 1 bins."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .run_stage3_dynamic_support import (
        BIN_SECONDS, ONSET_RULES, REGIONS, REVISION, SHOTS,
        extract_vectors, recording_vector,
    )
    from .run_stage2b_participant_residual import bootstrap_mean
except ImportError:
    from run_stage3_dynamic_support import (
        BIN_SECONDS, ONSET_RULES, REGIONS, REVISION, SHOTS,
        extract_vectors, recording_vector,
    )
    from run_stage2b_participant_residual import bootstrap_mean


TARGETS = ("share_change", "log_region_change", "log_total_change")
SUPPORT_WEIGHTS = {"actor_history": 0.0, "quarter_support": 0.25, "full_support": 1.0}
SPLIT_SEED = 20261127


def episode_targets(episode: dict, onset_rule: str) -> tuple[dict | None, str | None]:
    share, reason = recording_vector(episode, onset_rule)
    if reason is not None:
        return None, reason
    cutoff = round(share["cutoff_seconds_from_recording_start"] / BIN_SECONDS)
    trace = {
        round(float(row["time_bin_start_seconds"]) / BIN_SECONDS): row
        for row in episode["time_binned_pressure_trace_0p5s"]
        if row.get("time_bin_start_seconds") is not None
        and np.isfinite(float(row["time_bin_start_seconds"]))
        and abs(float(row["time_bin_start_seconds"]) / BIN_SECONDS
                - round(float(row["time_bin_start_seconds"]) / BIN_SECONDS)) < 1e-6
    }
    loads = np.asarray([
        [trace[index]["region_load_mean_N"][region] for region in REGIONS]
        for index in range(cutoff - 2, cutoff + 4)
    ], dtype=np.float64)
    if loads.shape != (6, 6) or not np.isfinite(loads).all() or (loads < 0).any():
        return None, "invalid_region_load_bin"
    log_region = np.log1p(loads)
    log_total = np.log1p(loads.sum(axis=1, keepdims=True))
    return {
        "share_change": share["delta"][:, :6],
        "log_region_change": log_region[2:] - log_region[1][None, :],
        "log_total_change": log_total[2:] - log_total[1][None, :],
    }, None


def extract_targets(stage1_report: dict, onset_rule: str) -> tuple[dict, dict]:
    stage3, stage3_inventory = extract_vectors(stage1_report, onset_rule)
    objects = {}
    for obj in stage1_report["objects"]:
        participants = defaultdict(list)
        for episode in obj["episodes"]:
            targets, reason = episode_targets(episode, onset_rule)
            if reason is None:
                participants[episode["subject_id"]].append(targets)
        if obj["object_name"] in stage3:
            objects[obj["object_name"]] = {
                actor: {
                    target: np.mean([record[target] for record in records], axis=0)
                    for target in TARGETS
                }
                for actor, records in participants.items()
            }
            if set(objects[obj["object_name"]]) != set(stage3[obj["object_name"]]):
                raise ValueError("Stage 4 eligibility differs from frozen Stage 3")
    if set(objects) != set(stage3):
        raise ValueError("Stage 4 object set differs from frozen Stage 3")
    return objects, stage3_inventory


def fit_sse(design: np.ndarray, response: np.ndarray) -> float:
    coefficient = np.linalg.lstsq(design, response, rcond=None)[0]
    return float(np.sum((response - design @ coefficient) ** 2))


def variance_decomposition(data: dict, target: str) -> dict:
    cells = [(name, actor, row[target].reshape(-1))
             for name, people in sorted(data.items()) for actor, row in sorted(people.items())]
    names = sorted(data)
    actors = sorted({actor for _, actor, _ in cells})
    response = np.stack([values for _, _, values in cells])
    object_design = np.asarray([[float(name == candidate) for candidate in names]
                                for name, _, _ in cells])
    actor_design = np.asarray([[float(actor == candidate) for candidate in actors]
                               for _, actor, _ in cells])
    total = float(np.sum((response - response.mean(axis=0)) ** 2))
    if total <= 0:
        raise ValueError("Target has no variation")
    object_sse = fit_sse(object_design, response)
    actor_sse = fit_sse(actor_design, response)
    additive_sse = fit_sse(np.column_stack((object_design, actor_design[:, 1:])), response)
    return {
        "cells": len(cells), "objects": len(names), "participants": len(actors),
        "dimensions": response.shape[1],
        "object_only_r2": 1 - object_sse / total,
        "participant_only_r2": 1 - actor_sse / total,
        "additive_r2": 1 - additive_sse / total,
        "incremental_object_after_participant_fraction": (actor_sse - additive_sse) / total,
        "incremental_participant_after_object_fraction": (object_sse - additive_sse) / total,
        "unexplained_fraction": additive_sse / total,
    }


def fixed_subsets(object_name: str, query_actor: str, support_actors: list[str],
                  shot: int, count: int, seed: int) -> list[tuple[str, ...]]:
    subsets = list(itertools.combinations(sorted(support_actors), shot))
    if not subsets:
        return []
    digest = hashlib.sha256(f"{object_name}|{query_actor}|{shot}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    indices = rng.choice(len(subsets), size=min(count, len(subsets)), replace=False)
    return [subsets[int(index)] for index in indices]


def cross_actor_prediction(data: dict, target: str, subsets_per_query: int,
                           repetitions: int, seed: int) -> tuple[dict, dict]:
    names = sorted(data)
    rows = {str(shot): [] for shot in SHOTS}
    for name in names:
        people = data[name]
        other_names = [candidate for candidate in names if candidate != name]
        histories = {
            actor: np.mean([data[other][actor][target] for other in other_names
                            if actor in data[other]], axis=0)
            for actor in people
        }
        effects = {actor: people[actor][target] - histories[actor] for actor in people}
        for shot in SHOTS:
            queries = []
            for actor, query in sorted(people.items()):
                support = [candidate for candidate in people if candidate != actor]
                subsets = fixed_subsets(name, actor, support, shot,
                                        subsets_per_query, SPLIT_SEED)
                if not subsets:
                    continue
                baseline = float(np.mean(np.abs(query[target] - histories[actor])))
                errors = {method: [] for method in SUPPORT_WEIGHTS if method != "actor_history"}
                for subset in subsets:
                    correction = np.mean([effects[candidate] for candidate in subset], axis=0)
                    for method, weight in SUPPORT_WEIGHTS.items():
                        if method != "actor_history":
                            prediction = histories[actor] + weight * correction
                            errors[method].append(float(np.mean(np.abs(query[target] - prediction))))
                queries.append({"actor_history": baseline,
                                **{method: float(np.mean(values)) for method, values in errors.items()}})
            rows[str(shot)].append({
                "object_name": name, "query_participants": len(queries),
                "metrics": {method: float(np.mean([query[method] for query in queries]))
                            for method in SUPPORT_WEIGHTS},
            })
    rng = np.random.default_rng(seed)
    summaries = {}
    for shot in SHOTS:
        values = rows[str(shot)]
        methods = {method: bootstrap_mean([row["metrics"][method] for row in values],
                                          rng, repetitions) for method in SUPPORT_WEIGHTS}
        paired = {}
        for method in ("quarter_support", "full_support"):
            differences = [row["metrics"]["actor_history"] - row["metrics"][method]
                           for row in values]
            paired[f"actor_history_minus_{method}"] = {
                **bootstrap_mean(differences, rng, repetitions),
                "objects_improved": int(np.sum(np.asarray(differences) > 0)),
            }
        summaries[str(shot)] = {"object_count": len(values), "methods": methods,
                                "paired_differences_positive_favors_support": paired}
    return summaries, rows


def run(source: dict, repetitions: int = 5000, subsets_per_query: int = 5,
        seed: int = 20261127) -> dict:
    if source.get("revision") != REVISION or source.get("object_count") != 63:
        raise ValueError("Expected frozen full Stage 1 report")
    result = {"stage": "Stage 4 exploratory object signal audit",
              "dataset_revision": REVISION, "seed": seed,
              "support_subset_seed": SPLIT_SEED,
              "bootstrap_repetitions": repetitions,
              "subsets_per_query": subsets_per_query,
              "onset_rules": {}}
    for rule in ONSET_RULES:
        data, inventory = extract_targets(source, rule)
        targets = {}
        for target in TARGETS:
            summaries, rows = cross_actor_prediction(data, target,
                                                      subsets_per_query, repetitions, seed)
            targets[target] = {
                "variance_decomposition": variance_decomposition(data, target),
                "shot_summaries": summaries,
                "object_metrics": rows,
            }
        result["onset_rules"][rule] = {"inventory": inventory, "targets": targets}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--subsets-per-query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = run(source, args.bootstrap_reps, args.subsets_per_query, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    for rule, report in result["onset_rules"].items():
        print(rule, flush=True)
        for target, audit in report["targets"].items():
            v = audit["variance_decomposition"]
            s = audit["shot_summaries"]["5"]
            gain = s["paired_differences_positive_favors_support"]["actor_history_minus_quarter_support"]
            print(target, "object_partial_r2", round(v["incremental_object_after_participant_fraction"], 6),
                  "actor_r2", round(v["participant_only_r2"], 6),
                  "actor_mae", round(s["methods"]["actor_history"]["mean"], 6),
                  "five_shot_quarter_gain", round(gain["mean"], 6),
                  "ci95", [round(x, 6) for x in gain["ci95"]],
                  "objects_improved", gain["objects_improved"], flush=True)


if __name__ == "__main__":
    main()

