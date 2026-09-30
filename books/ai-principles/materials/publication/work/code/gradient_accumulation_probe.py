"""C20: compare one large batch with four gradient-accumulation microbatches.

The 64 training blocks deliberately mix full and padded tail blocks. A microbatch
mean must be weighted by its real target count; otherwise it changes the
objective. No optimizer step or held-out text is used here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from train_causal_wikitext import (
    WORK, TrainConfig, batch, create, environment, load_manifest, load_split,
    loss_sum, write_json,
)


def gradients(model) -> list[torch.Tensor]:
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def compare(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict:
    max_abs = max(float((a - b).abs().max()) for a, b in zip(left, right))
    difference_sq = sum(float(((a - b).float() ** 2).sum()) for a, b in zip(left, right))
    reference_sq = sum(float((a.float() ** 2).sum()) for a in left)
    return {
        "max_absolute_gradient_difference": max_abs,
        "relative_gradient_l2_difference": math.sqrt(difference_sq / reference_sq),
    }


def compute(model, part, groups, device, mode, total_targets):
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    individual = []
    total_loss = 0.0
    for selected in groups:
        ids, labels, real = batch(part, selected, device)
        logits = model(ids, real)
        sum_loss, count = loss_sum(logits, labels)
        if mode == "weighted":
            objective = sum_loss / total_targets
        elif mode == "naive":
            objective = (sum_loss / count) / len(groups)
        else:
            objective = sum_loss / count
        objective.backward()
        individual.append(count)
        total_loss += float(sum_loss.detach())
        del ids, labels, real, logits, sum_loss, objective
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "gradients": gradients(model),
        "microbatch_target_counts": individual,
        "reference_target_nll": total_loss / total_targets,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK / "results" / "c20_accumulation.json")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else WORK.parent / args.out
    if out.exists():
        raise FileExistsError(f"keep previous result; choose another --out: {out}")
    manifest, manifest_sha = load_manifest()
    part = load_split(manifest, "train")
    counts = (part["labels"] != -100).sum(axis=1)
    if not np.array_equal(counts, part["attention_mask"].sum(axis=1)):
        raise AssertionError("当前C18纯右邻块的有效标签与真实输入位置应一一对应")
    full = np.flatnonzero(counts == 128)[:32]
    tails = np.flatnonzero(counts < 128)[:32]
    if len(full) != 32 or len(tails) != 32:
        raise ValueError("expected both full and tail training blocks")
    selected = np.concatenate([full, tails])
    groups = [selected[i:i + 16] for i in range(0, 64, 16)]
    total_targets = int(counts[selected].sum())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260924)
    model, _ = create(TrainConfig(), device)
    full_run = compute(model, part, [selected], device, "full", total_targets)
    weighted = compute(model, part, groups, device, "weighted", total_targets)
    naive = compute(model, part, groups, device, "naive", total_targets)
    output = {
        "scope": "64 C18 training blocks; 32 full and 32 padded tails; random C19 model; gradients only, no optimizer step or validation/test",
        "manifest_sha256": manifest_sha,
        "environment": environment(device),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "full_batch_target_count": total_targets,
        "counted_target_rule": "labels != -100; current C18 pure next-token blocks happen to match readable input counts",
        "microbatch_target_counts": weighted["microbatch_target_counts"],
        "full_batch_nll": full_run["reference_target_nll"],
        "weighted_against_full": compare(full_run["gradients"], weighted["gradients"]),
        "naive_equal_micro_means_against_full": compare(
            full_run["gradients"], naive["gradients"]
        ),
        "peak_allocated_bytes": {
            "full_batch": full_run["peak_allocated_bytes"],
            "weighted_microbatches": weighted["peak_allocated_bytes"],
            "naive_microbatches": naive["peak_allocated_bytes"],
        },
        "equation": "sum_i grad(sum_loss_i / sum_j N_j), versus mean_i grad(sum_loss_i / N_i)",
    }
    write_json(out, output)
    print(
        f"targets={total_targets} groups={weighted['microbatch_target_counts']} "
        f"weighted_relative={output['weighted_against_full']['relative_gradient_l2_difference']:.3g} "
        f"naive_relative={output['naive_equal_micro_means_against_full']['relative_gradient_l2_difference']:.3g}"
    )


if __name__ == "__main__":
    main()
