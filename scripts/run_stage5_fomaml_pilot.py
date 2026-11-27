#!/usr/bin/env python3
"""Exploratory first-order MAML pilot on held-out-object pressure trajectories."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_stage4_log_region_model_pilot import (
    ONSET_RULE, SPLIT_SEED, TARGET, evaluate_fold, make_model,
    supervised_train, step,
)
from run_stage4_object_signal_audit import extract_targets
from run_stage3_model_pilot import object_folds
from run_stage3_dynamic_support import REVISION, SHOTS
from run_stage2b_participant_residual import bootstrap_mean


METHODS = (
    "persistence", "global_transition", "actor_transition",
    "actor_quarter_support", "mlp_pretrained", "mlp_finetuned",
    "fomaml_pretrained", "fomaml_finetuned",
)


def choose_task(data, train_names, rng, query_size):
    """Sample an object task with disjoint participant-level Support and Query."""
    name = train_names[int(rng.integers(len(train_names)))]
    actors = sorted(data[name])
    shot = int(rng.choice(SHOTS))
    if len(actors) <= shot:
        raise ValueError(f"Object {name} has too few participants for {shot}-shot")
    chosen = rng.choice(len(actors), size=shot + min(query_size, len(actors) - shot),
                        replace=False)
    support_actors = [actors[i] for i in chosen[:shot]]
    query_actors = [actors[i] for i in chosen[shot:]]
    return (name, support_actors, query_actors)


def tensors(data, name, actors):
    records = [data[name][actor] for actor in actors]
    x = torch.tensor(np.stack([row["early_log_region"] for row in records]),
                     dtype=torch.float32)
    y = torch.tensor(np.stack([row[TARGET] for row in records]),
                     dtype=torch.float32)
    return x, y


def fomaml_train(data, train_names, seed, iterations, inner_steps, inner_lr,
                 outer_lr, query_size):
    """First-order MAML: query gradients at adapted weights update initialization."""
    model = make_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    rng = np.random.default_rng(seed + 200000)
    recent_query_losses = []
    shot_counts = {str(shot): 0 for shot in SHOTS}
    for iteration in range(iterations):
        name, support_actors, query_actors = choose_task(
            data, train_names, rng, query_size)
        shot_counts[str(len(support_actors))] += 1
        sx, sy = tensors(data, name, support_actors)
        qx, qy = tensors(data, name, query_actors)
        fast = copy.deepcopy(model)
        inner_optimizer = torch.optim.SGD(fast.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            step(fast, inner_optimizer, sx, sy)
        fast.zero_grad(set_to_none=True)
        query_loss = torch.mean((fast(qx) - qy) ** 2)
        query_loss.backward()
        nn.utils.clip_grad_norm_(fast.parameters(), 1.0)
        optimizer.zero_grad(set_to_none=True)
        for parameter, adapted in zip(model.parameters(), fast.parameters()):
            parameter.grad = adapted.grad.detach().clone()
        optimizer.step()
        if iteration >= max(0, iterations - 100):
            recent_query_losses.append(float(query_loss.detach()))
    return model, {
        "last_100_training_query_mse": float(np.mean(recent_query_losses)),
        "sampled_support_shots": shot_counts,
    }


def summarize(rows, repetitions, seed):
    rng = np.random.default_rng(seed)
    comparisons = (
        ("mlp_pretrained", "mlp_finetuned"),
        ("fomaml_pretrained", "fomaml_finetuned"),
        ("mlp_finetuned", "fomaml_finetuned"),
        ("actor_quarter_support", "fomaml_finetuned"),
    )
    output = {}
    for shot in SHOTS:
        values = rows[str(shot)]
        methods = {method: bootstrap_mean(
            [row["metrics"][method] for row in values], rng, repetitions)
            for method in METHODS}
        paired = {}
        for first, second in comparisons:
            differences = np.asarray([
                row["metrics"][first] - row["metrics"][second]
                for row in values])
            paired[f"{first}_minus_{second}"] = {
                **bootstrap_mean(differences, rng, repetitions),
                "objects_improved": int(np.sum(differences > 0)),
            }
        output[str(shot)] = {
            "object_count": len(values), "methods": methods,
            "paired_differences_positive_favors_second": paired,
        }
    return output


def run(source, *, seed=20261127, fold_count=5, fold_limit=None,
        supervised_steps=1500, meta_iterations=1500, inner_steps=3,
        inner_lr=0.05, outer_lr=0.001, mlp_adapt_lr=0.05,
        fomaml_adapt_lr=0.05, batch_size=3, query_size=3,
        subsets_per_query=5, bootstrap_reps=5000):
    if source.get("revision") != REVISION or source.get("object_count") != 63:
        raise ValueError("Expected frozen full Stage 1 report")
    torch.set_num_threads(1)
    data, inventory = extract_targets(source, ONSET_RULE)
    names = sorted(data)
    folds = object_folds(names, SPLIT_SEED, fold_count)
    selected_folds = folds if fold_limit is None else folds[:fold_limit]
    all_rows = {str(shot): [] for shot in SHOTS}
    diagnostics = []
    for fold_index, test_names in enumerate(selected_folds):
        train_names = [name for name in names if name not in set(test_names)]
        model_seed = seed + fold_index
        mlp = supervised_train(data, train_names, model_seed,
                               supervised_steps, batch_size)
        fomaml, diagnostic = fomaml_train(
            data, train_names, model_seed, meta_iterations, inner_steps,
            inner_lr, outer_lr, query_size)
        diagnostics.append({"fold": fold_index, **diagnostic})
        fold_rows = evaluate_fold(
            data, train_names, test_names, mlp, fomaml, inner_steps,
            mlp_adapt_lr, fomaml_adapt_lr, subsets_per_query)
        for shot in SHOTS:
            for row in fold_rows[str(shot)]:
                row["metrics"]["fomaml_pretrained"] = row["metrics"].pop(
                    "reptile_pretrained")
                row["metrics"]["fomaml_finetuned"] = row["metrics"].pop(
                    "reptile_finetuned")
                all_rows[str(shot)].append(row)
        print(f"completed fold {fold_index + 1}/{len(selected_folds)}",
              flush=True)
    return {
        "stage": "Stage 5 exploratory FOMAML pilot",
        "dataset_revision": REVISION, "onset_rule": ONSET_RULE,
        "target": TARGET, "scenario": "held-out object, known participant",
        "folds": folds, "folds_completed": len(selected_folds),
        "inventory": inventory,
        "hyperparameters": {
            "seed": seed, "fold_count": fold_count,
            "split_and_support_subset_seed": SPLIT_SEED,
            "supervised_steps": supervised_steps,
            "meta_iterations": meta_iterations,
            "inner_steps": inner_steps, "inner_lr": inner_lr,
            "outer_lr": outer_lr,
            "mlp_adapt_lr": mlp_adapt_lr,
            "fomaml_adapt_lr": fomaml_adapt_lr,
            "batch_size": batch_size, "query_size": query_size,
            "subsets_per_query": subsets_per_query,
            "bootstrap_repetitions": bootstrap_reps,
        },
        "training_diagnostics": diagnostics,
        "shot_summaries": summarize(all_rows, bootstrap_reps, seed),
        "object_metrics": all_rows,
        "status": "exploratory; object set already inspected during Stage 3/4",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--supervised-steps", type=int, default=1500)
    parser.add_argument("--meta-iterations", type=int, default=1500)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--outer-lr", type=float, default=0.001)
    parser.add_argument("--mlp-adapt-lr", type=float, default=0.05)
    parser.add_argument("--fomaml-adapt-lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--query-size", type=int, default=3)
    parser.add_argument("--subsets-per-query", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = run(source, seed=args.seed, fold_count=args.folds,
                 fold_limit=args.fold_limit,
                 supervised_steps=args.supervised_steps,
                 meta_iterations=args.meta_iterations,
                 inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                 outer_lr=args.outer_lr, mlp_adapt_lr=args.mlp_adapt_lr,
                 fomaml_adapt_lr=args.fomaml_adapt_lr,
                 batch_size=args.batch_size, query_size=args.query_size,
                 subsets_per_query=args.subsets_per_query,
                 bootstrap_reps=args.bootstrap_reps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(result["shot_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()

