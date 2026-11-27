#!/usr/bin/env python3
"""Exploratory five-fold MLP–Reptile pilot on log regional pressure change."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .run_stage4_object_signal_audit import extract_targets
    from .run_stage3_model_pilot import fixed_subsets, object_folds
    from .run_stage3_dynamic_support import REVISION, SHOTS
    from .run_stage2b_participant_residual import bootstrap_mean, global_mean, mae
except ImportError:
    from run_stage4_object_signal_audit import extract_targets
    from run_stage3_model_pilot import fixed_subsets, object_folds
    from run_stage3_dynamic_support import REVISION, SHOTS
    from run_stage2b_participant_residual import bootstrap_mean, global_mean, mae


ONSET_RULE = "two_sensors_gt_5N_3samples"
TARGET = "log_region_change"
SPLIT_SEED = 20261127
LOSS_SCALE = 1.0
METHODS = ("persistence", "global_transition", "actor_transition",
           "actor_quarter_support", "mlp_pretrained", "mlp_finetuned",
           "reptile_pretrained", "reptile_finetuned")


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(12, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 24))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, early):
        return self.layers(early.reshape(-1, 12)).reshape(-1, 4, 6)


def make_model(seed):
    torch.manual_seed(seed)
    return ResidualMLP()


def examples(data, train_names):
    records = [record for name in train_names for record in data[name].values()]
    x = torch.tensor(np.stack([record["early_log_region"] for record in records]),
                     dtype=torch.float32)
    y = torch.tensor(np.stack([record[TARGET] for record in records]),
                     dtype=torch.float32)
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
        actors = sorted(data[name])
        shot = int(rng.choice(SHOTS))
        selected = rng.choice(actors, size=shot, replace=False)
        support = [data[name][actor] for actor in selected]
        x = torch.tensor(np.stack([record["early_log_region"] for record in support]),
                         dtype=torch.float32)
        y = torch.tensor(np.stack([record[TARGET] for record in support]),
                         dtype=torch.float32)
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
    x = torch.tensor(query["early_log_region"][None, :, :], dtype=torch.float32)
    with torch.no_grad():
        return model(x)[0].numpy()


def adapted_predict(model, support, query, inner_steps, inner_lr):
    fast = copy.deepcopy(model)
    x = torch.tensor(np.stack([row["early_log_region"] for row in support]),
                     dtype=torch.float32)
    y = torch.tensor(np.stack([row[TARGET] for row in support]), dtype=torch.float32)
    optimizer = torch.optim.SGD(fast.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        step(fast, optimizer, x, y)
    return predict(fast, query)


def actor_history(data, train_names, actor):
    values = [data[name][actor][TARGET] for name in train_names if actor in data[name]]
    return np.mean(values, axis=0) if values else None


def evaluate_fold(data, train_names, test_names, mlp, reptile,
                  inner_steps, mlp_adapt_lr, reptile_adapt_lr, subset_count):
    global_delta = global_mean(data, train_names, TARGET)
    rows = {str(shot): [] for shot in SHOTS}
    for name in test_names:
        people = data[name]
        histories = {actor: actor_history(data, train_names, actor) for actor in people}
        if any(value is None for value in histories.values()):
            raise ValueError("A held-out participant has no training-object history")
        for shot in SHOTS:
            queries = []
            for actor, query in sorted(people.items()):
                target = query[TARGET]
                subsets = fixed_subsets(name, actor,
                                        [other for other in people if other != actor],
                                        shot, subset_count, SPLIT_SEED)
                fixed = {
                    "persistence": mae(target, np.zeros_like(target)),
                    "global_transition": mae(target, global_delta),
                    "actor_transition": mae(target, histories[actor]),
                    "mlp_pretrained": mae(target, predict(mlp, query)),
                    "reptile_pretrained": mae(target, predict(reptile, query)),
                }
                actor_errors, mlp_errors, reptile_errors = [], [], []
                for subset in subsets:
                    support = [people[other] for other in subset]
                    object_effect = np.mean([people[other][TARGET] - histories[other]
                                             for other in subset], axis=0)
                    actor_errors.append(mae(target, histories[actor] + 0.25 * object_effect))
                    mlp_errors.append(mae(target, adapted_predict(
                        mlp, support, query, inner_steps, mlp_adapt_lr)))
                    reptile_errors.append(mae(target, adapted_predict(
                        reptile, support, query, inner_steps, reptile_adapt_lr)))
                queries.append({**fixed,
                                "actor_quarter_support": float(np.mean(actor_errors)),
                                "mlp_finetuned": float(np.mean(mlp_errors)),
                                "reptile_finetuned": float(np.mean(reptile_errors))})
            rows[str(shot)].append({
                "object_name": name, "query_participants": len(queries),
                "metrics": {method: float(np.mean([query[method] for query in queries]))
                            for method in METHODS},
            })
    return rows


def summarize(rows, repetitions, seed):
    rng = np.random.default_rng(seed)
    result = {}
    comparisons = (("mlp_pretrained", "mlp_finetuned"),
                   ("reptile_pretrained", "reptile_finetuned"),
                   ("mlp_finetuned", "reptile_finetuned"),
                   ("actor_quarter_support", "mlp_finetuned"),
                   ("actor_quarter_support", "reptile_finetuned"))
    for shot in SHOTS:
        values = rows[str(shot)]
        methods = {method: bootstrap_mean([row["metrics"][method] for row in values],
                                          rng, repetitions) for method in METHODS}
        paired = {}
        for baseline, adapted in comparisons:
            differences = [row["metrics"][baseline] - row["metrics"][adapted]
                           for row in values]
            paired[f"{baseline}_minus_{adapted}"] = {
                **bootstrap_mean(differences, rng, repetitions),
                "objects_improved": int(np.sum(np.asarray(differences) > 0)),
            }
        result[str(shot)] = {"object_count": len(values), "methods": methods,
                             "paired_differences_positive_favors_second": paired}
    return result


def run(source, *, seed=20261127, fold_count=5, fold_limit=None,
        supervised_steps=1500, meta_iterations=500, inner_steps=3,
        inner_lr=0.05, mlp_adapt_lr=None, reptile_adapt_lr=None,
        batch_size=3, subsets_per_query=5,
        bootstrap_reps=5000):
    if source.get("revision") != REVISION or source.get("object_count") != 63:
        raise ValueError("Expected frozen full Stage 1 report")
    torch.set_num_threads(1)
    mlp_adapt_lr = inner_lr if mlp_adapt_lr is None else mlp_adapt_lr
    reptile_adapt_lr = inner_lr if reptile_adapt_lr is None else reptile_adapt_lr
    data, inventory = extract_targets(source, ONSET_RULE)
    names = sorted(data)
    folds = object_folds(names, SPLIT_SEED, fold_count)
    selected_folds = folds if fold_limit is None else folds[:fold_limit]
    all_rows = {str(shot): [] for shot in SHOTS}
    for fold_index, test_names in enumerate(selected_folds):
        train_names = [name for name in names if name not in set(test_names)]
        model_seed = seed + fold_index
        mlp = supervised_train(data, train_names, model_seed, supervised_steps, batch_size)
        reptile = reptile_train(data, train_names, model_seed,
                               meta_iterations, inner_steps, inner_lr)
        fold_rows = evaluate_fold(data, train_names, test_names, mlp, reptile,
                                  inner_steps, mlp_adapt_lr, reptile_adapt_lr,
                                  subsets_per_query)
        for shot in SHOTS:
            all_rows[str(shot)].extend(fold_rows[str(shot)])
        print(f"completed fold {fold_index + 1}/{len(selected_folds)} with {len(test_names)} objects",
              flush=True)
    return {
        "stage": "Stage 4 exploratory log-region MLP–Reptile pilot",
        "dataset_revision": REVISION, "onset_rule": ONSET_RULE,
        "target": TARGET, "scenario": "held-out object, known participant",
        "folds": folds, "folds_completed": len(selected_folds),
        "inventory": inventory,
        "hyperparameters": {
            "seed": seed, "fold_count": fold_count,
            "split_and_support_subset_seed": SPLIT_SEED,
            "supervised_steps": supervised_steps, "meta_iterations": meta_iterations,
            "inner_steps": inner_steps, "inner_lr": inner_lr,
            "mlp_adapt_lr": mlp_adapt_lr, "reptile_adapt_lr": reptile_adapt_lr,
            "batch_size": batch_size, "subsets_per_query": subsets_per_query,
            "loss_scale": LOSS_SCALE, "hidden_width": 32,
            "bootstrap_repetitions": bootstrap_reps,
        },
        "shot_summaries": summarize(all_rows, bootstrap_reps, seed),
        "object_metrics": all_rows,
        "status": "exploratory post hoc target change; no untouched test objects",
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
    parser.add_argument("--mlp-adapt-lr", type=float)
    parser.add_argument("--reptile-adapt-lr", type=float)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--subsets-per-query", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = run(source, seed=args.seed, fold_count=args.folds, fold_limit=args.fold_limit,
                 supervised_steps=args.supervised_steps, meta_iterations=args.meta_iterations,
                 inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                 mlp_adapt_lr=args.mlp_adapt_lr, reptile_adapt_lr=args.reptile_adapt_lr,
                 batch_size=args.batch_size,
                 subsets_per_query=args.subsets_per_query, bootstrap_reps=args.bootstrap_reps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(result["shot_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()

