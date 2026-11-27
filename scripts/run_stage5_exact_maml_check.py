#!/usr/bin/env python3
"""One-fold feasibility check for exact second-order MAML on the Stage 5 task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from run_stage5_fomaml_pilot import choose_task, tensors
from run_stage4_log_region_model_pilot import (
    ONSET_RULE, SPLIT_SEED, TARGET, evaluate_fold, make_model, supervised_train,
)
from run_stage4_object_signal_audit import extract_targets
from run_stage3_model_pilot import object_folds
from run_stage3_dynamic_support import REVISION


def maml_train(data, train_names, seed, iterations, inner_steps=3,
               inner_lr=0.05, outer_lr=0.001, query_size=3):
    model = make_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    rng = np.random.default_rng(seed + 200000)
    last_losses = []
    for iteration in range(iterations):
        name, support_actors, query_actors = choose_task(
            data, train_names, rng, query_size)
        sx, sy = tensors(data, name, support_actors)
        qx, qy = tensors(data, name, query_actors)
        fast = dict(model.named_parameters())
        for _ in range(inner_steps):
            support_loss = torch.mean((functional_call(model, fast, (sx,)) - sy) ** 2)
            gradients = torch.autograd.grad(
                support_loss, tuple(fast.values()), create_graph=True)
            norm = torch.sqrt(sum(torch.sum(gradient.square()) for gradient in gradients))
            scale = torch.clamp(1.0 / (norm + 1e-12), max=1.0)
            fast = {key: value - inner_lr * scale * gradient
                    for (key, value), gradient in zip(fast.items(), gradients)}
        query_loss = torch.mean((functional_call(model, fast, (qx,)) - qy) ** 2)
        optimizer.zero_grad(set_to_none=True)
        query_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if iteration >= max(0, iterations - 100):
            last_losses.append(float(query_loss.detach()))
    return model, float(np.mean(last_losses))


def run(source, seed=20261127, iterations=1500, fold_index=0,
        subsets_per_query=5):
    if source.get("revision") != REVISION or source.get("object_count") != 63:
        raise ValueError("Expected frozen Stage 1 report")
    torch.set_num_threads(1)
    data, _ = extract_targets(source, ONSET_RULE)
    names = sorted(data)
    folds = object_folds(names, SPLIT_SEED, 5)
    test_names = folds[fold_index]
    train_names = [name for name in names if name not in set(test_names)]
    model_seed = seed + fold_index
    mlp = supervised_train(data, train_names, model_seed, 1500, 3)
    maml, training_loss = maml_train(data, train_names, model_seed, iterations)
    rows = evaluate_fold(data, train_names, test_names, mlp, maml,
                         3, 0.05, 0.05, subsets_per_query)
    output = {"stage": "Stage 5 exact MAML one-fold feasibility check",
              "seed": seed, "fold_index": fold_index,
              "test_objects": test_names, "meta_iterations": iterations,
              "last_100_training_query_mse": training_loss,
              "status": "exploratory inspected fold; no independent model selection"}
    for shot, values in rows.items():
        output[shot] = {
            "mlp_pretrained": float(np.mean([v["metrics"]["mlp_pretrained"] for v in values])),
            "mlp_finetuned": float(np.mean([v["metrics"]["mlp_finetuned"] for v in values])),
            "maml_pretrained": float(np.mean([v["metrics"]["reptile_pretrained"] for v in values])),
            "maml_finetuned": float(np.mean([v["metrics"]["reptile_finetuned"] for v in values])),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--subsets-per-query", type=int, default=5)
    args = parser.parse_args()
    result = run(json.loads(args.input.read_text(encoding="utf-8")),
                 seed=args.seed, iterations=args.iterations,
                 fold_index=args.fold_index,
                 subsets_per_query=args.subsets_per_query)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

