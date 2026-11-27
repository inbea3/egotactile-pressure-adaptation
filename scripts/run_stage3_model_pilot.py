#!/usr/bin/env python3
"""Bounded, exploratory MLP versus Reptile pilot on Stage 3 pressure bins.

Object folds and Support subsets are shared across methods. The model target is
four future seven-channel contact rows; each input has two observed rows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .run_stage3_dynamic_support import REVISION, extract_vectors, project_rows
    from .run_stage2b_participant_residual import bootstrap_mean, global_mean, mae
except ImportError:
    from run_stage3_dynamic_support import REVISION, extract_vectors, project_rows
    from run_stage2b_participant_residual import bootstrap_mean, global_mean, mae


ONSET_RULE = "two_sensors_gt_5N_3samples"
SHOTS = (1, 2, 5)
METHODS = ("persistence", "global_transition", "actor_transition",
           "mlp_pretrained", "mlp_finetuned", "reptile_pretrained", "reptile_finetuned")
LOSS_SCALE = 100.0
SPLIT_SEED = 20261127


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(14, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 28))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, early):
        return self.layers(early.reshape(-1, 14)).reshape(-1, 4, 7)


def make_model(seed):
    torch.manual_seed(seed)
    return ResidualMLP()


def examples(data, object_names):
    pairs = [row for name in object_names for row in data[name].values()]
    x = torch.tensor(np.stack([row["early"] for row in pairs]), dtype=torch.float32)
    y = torch.tensor(np.stack([row["delta"] for row in pairs]), dtype=torch.float32)
    return x, y


def step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = LOSS_SCALE * torch.mean((model(x) - y) ** 2)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def supervised_train(data, train_names, seed, steps, batch_size):
    model = make_model(seed)
    x, y = examples(data, train_names)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        indices = rng.integers(0, len(x), size=min(batch_size, len(x)))
        step(model, optimizer, x[indices], y[indices])
    return model


def reptile_train(data, train_names, seed, iterations, inner_steps, inner_lr):
    model = make_model(seed)
    rng = np.random.default_rng(seed + 100000)
    for iteration in range(iterations):
        name = train_names[int(rng.integers(len(train_names)))]
        people = sorted(data[name])
        shot = int(rng.choice(SHOTS))
        selected = rng.choice(people, size=shot, replace=False)
        support = [data[name][subject] for subject in selected]
        x = torch.tensor(np.stack([row["early"] for row in support]), dtype=torch.float32)
        y = torch.tensor(np.stack([row["delta"] for row in support]), dtype=torch.float32)
        fast = copy.deepcopy(model)
        optimizer = torch.optim.SGD(fast.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            step(fast, optimizer, x, y)
        alpha = 0.1 - 0.09 * iteration / max(1, iterations - 1)
        with torch.no_grad():
            for parameter, adapted in zip(model.parameters(), fast.parameters()):
                parameter.add_(alpha * (adapted - parameter))
    return model


def predict(model, query):
    x = torch.tensor(query["early"][None, :, :], dtype=torch.float32)
    with torch.no_grad():
        correction = model(x)[0].numpy()
    persistence = np.repeat(query["early"][-1][None, :], 4, axis=0)
    return project_rows(persistence + correction)


def adapted_predict(model, support, query, inner_steps, inner_lr):
    fast = copy.deepcopy(model)
    x = torch.tensor(np.stack([row["early"] for row in support]), dtype=torch.float32)
    y = torch.tensor(np.stack([row["delta"] for row in support]), dtype=torch.float32)
    optimizer = torch.optim.SGD(fast.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        step(fast, optimizer, x, y)
    return predict(fast, query)


def fixed_subsets(object_name, query_subject, support_subjects, shot, count, seed):
    subsets = list(itertools.combinations(sorted(support_subjects), shot))
    digest = hashlib.sha256(f"{object_name}|{query_subject}|{shot}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    indices = rng.choice(len(subsets), size=min(count, len(subsets)), replace=False)
    return [subsets[int(index)] for index in indices]


def object_folds(names, seed, fold_count):
    order = np.random.default_rng(seed).permutation(sorted(names)).tolist()
    return [sorted(order[index::fold_count]) for index in range(fold_count)]


def evaluate_fold(data, train_names, test_names, mlp, reptile, inner_steps, inner_lr, subset_count, subset_seed):
    global_delta = global_mean(data, train_names, "delta")
    rows = {str(shot): [] for shot in SHOTS}
    for name in test_names:
        people = data[name]
        for shot in SHOTS:
            query_metrics = []
            for subject, query in sorted(people.items()):
                support_subjects = [other for other in people if other != subject]
                subsets = fixed_subsets(name, subject, support_subjects, shot, subset_count, subset_seed)
                persistence = np.repeat(query["early"][-1][None, :], 4, axis=0)
                actor_values = [data[other][subject]["delta"] for other in train_names if subject in data[other]]
                actor_delta = np.mean(actor_values, axis=0) if actor_values else None
                fixed_errors = {
                    "persistence": mae(query["future"], persistence),
                    "global_transition": mae(query["future"], project_rows(persistence + global_delta)),
                    "actor_transition": mae(query["future"], project_rows(persistence + actor_delta))
                    if actor_delta is not None else None,
                    "mlp_pretrained": mae(query["future"], predict(mlp, query)),
                    "reptile_pretrained": mae(query["future"], predict(reptile, query)),
                }
                mlp_errors, reptile_errors = [], []
                for subset in subsets:
                    support = [people[other] for other in subset]
                    mlp_errors.append(mae(query["future"], adapted_predict(mlp, support, query, inner_steps, inner_lr)))
                    reptile_errors.append(mae(query["future"], adapted_predict(reptile, support, query, inner_steps, inner_lr)))
                query_metrics.append({**fixed_errors,
                                      "mlp_finetuned": float(np.mean(mlp_errors)),
                                      "reptile_finetuned": float(np.mean(reptile_errors))})
            rows[str(shot)].append({
                "object_name": name,
                "query_participants": len(query_metrics),
                "metrics": {
                    method: float(np.mean([row[method] for row in query_metrics if row[method] is not None]))
                    if any(row[method] is not None for row in query_metrics) else None
                    for method in METHODS
                },
            })
    return rows


def summarize(rows, repetitions, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for shot in SHOTS:
        values = rows[str(shot)]
        methods = {
            method: bootstrap_mean([row["metrics"][method] for row in values], rng, repetitions)
            for method in METHODS if all(row["metrics"][method] is not None for row in values)
        }
        comparisons = {}
        for baseline, adapted in (("mlp_pretrained", "mlp_finetuned"),
                                  ("mlp_finetuned", "reptile_finetuned"),
                                  ("actor_transition", "reptile_finetuned"),
                                  ("global_transition", "reptile_finetuned")):
            if baseline not in methods or adapted not in methods:
                continue
            differences = [row["metrics"][baseline] - row["metrics"][adapted] for row in values]
            comparisons[f"{baseline}_minus_{adapted}"] = {
                **bootstrap_mean(differences, rng, repetitions),
                "objects_improved": int(np.sum(np.asarray(differences) > 0)),
            }
        result[str(shot)] = {"object_count": len(values), "methods": methods,
                             "paired_differences_positive_favors_second": comparisons}
    return result


def run(source, *, seed=20261127, fold_count=5, fold_limit=None, supervised_steps=1500,
        meta_iterations=500, inner_steps=3, inner_lr=0.05, batch_size=32,
        subsets_per_query=5, bootstrap_reps=5000):
    if source.get("revision") != REVISION or source.get("object_count") != 63:
        raise ValueError("Expected frozen full Stage 1 report")
    torch.set_num_threads(1)
    data, inventory = extract_vectors(source, ONSET_RULE)
    names = sorted(data)
    folds = object_folds(names, SPLIT_SEED, fold_count)
    selected_folds = folds if fold_limit is None else folds[:fold_limit]
    if any(len(fold) == 0 for fold in selected_folds):
        raise ValueError("Empty test fold")
    all_rows = {str(shot): [] for shot in SHOTS}
    for fold_index, test_names in enumerate(selected_folds):
        train_names = [name for name in names if name not in set(test_names)]
        model_seed = seed + fold_index
        mlp = supervised_train(data, train_names, model_seed, supervised_steps, batch_size)
        reptile = reptile_train(data, train_names, model_seed, meta_iterations, inner_steps, inner_lr)
        fold_rows = evaluate_fold(data, train_names, test_names, mlp, reptile,
                                  inner_steps, inner_lr, subsets_per_query, SPLIT_SEED)
        for shot in SHOTS:
            all_rows[str(shot)].extend(fold_rows[str(shot)])
        print(f"completed fold {fold_index + 1}/{len(selected_folds)} with {len(test_names)} objects", flush=True)
    return {
        "stage": "Stage 3 bounded exploratory MLP and Reptile pilot",
        "dataset_revision": REVISION,
        "onset_rule": ONSET_RULE,
        "scenario": "held-out object, known participants; same-object Support excludes Query participant",
        "folds": folds,
        "folds_completed": len(selected_folds),
        "inventory": inventory,
        "hyperparameters": {
            "seed": seed, "fold_count": fold_count, "supervised_steps": supervised_steps,
            "split_and_support_subset_seed": SPLIT_SEED,
            "meta_iterations": meta_iterations, "inner_steps": inner_steps,
            "inner_lr": inner_lr, "batch_size": batch_size,
            "subsets_per_query": subsets_per_query, "loss_scale": LOSS_SCALE,
            "hidden_width": 32, "bootstrap_repetitions": bootstrap_reps,
        },
        "shot_summaries": summarize(all_rows, bootstrap_reps, seed),
        "object_metrics": all_rows,
        "status": "exploratory pilot; Stage 3 no-model gate was not met",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--supervised-steps", type=int, default=1500)
    parser.add_argument("--meta-iterations", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--subsets-per-query", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = run(source, seed=args.seed, fold_count=args.folds, fold_limit=args.fold_limit,
                 supervised_steps=args.supervised_steps, meta_iterations=args.meta_iterations,
                 inner_steps=args.inner_steps, inner_lr=args.inner_lr, batch_size=args.batch_size,
                 subsets_per_query=args.subsets_per_query, bootstrap_reps=args.bootstrap_reps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["shot_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()

