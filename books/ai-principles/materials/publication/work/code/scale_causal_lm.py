"""C20: measure C19's training resources and a narrow depth comparison.

Run from the package root:
    python work/code/scale_causal_lm.py profile
    python work/code/scale_causal_lm.py depth

Only the C18 train/validation arrays are opened. This experiment never
selects a model by the held-out test, edits C19 results, or claims a scaling law.
"""

from __future__ import annotations

import argparse
import gc
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from train_causal_wikitext import (
    DATA, WORK, TrainConfig, autocast_for, batch, create, environment,
    evaluate_loss, fixed_train_indices, learning_rate, load_manifest,
    load_split, loss_sum, set_lr, sha256, write_json,
)


def tensor_bytes(values) -> int:
    if isinstance(values, torch.Tensor):
        return values.numel() * values.element_size()
    if isinstance(values, dict):
        return sum(tensor_bytes(value) for value in values.values())
    if isinstance(values, (list, tuple)):
        return sum(tensor_bytes(value) for value in values)
    return 0


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_update(model, optimizer, ids, labels, real, device, cfg, step, total_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    set_lr(optimizer, learning_rate(step, total_steps, cfg))
    with autocast_for(device):
        logits = model(ids, real)
    summed, count = loss_sum(logits, labels)
    loss = summed / count
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise ValueError("nonfinite gradient")
    optimizer.step()
    return float(summed.detach()), count


def profile(out: Path, warmup: int, measured: int) -> None:
    if out.exists():
        raise FileExistsError(f"keep previous measurement; choose another --out: {out}")
    manifest, manifest_sha = load_manifest()
    train = load_split(manifest, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = []
    for batch_size in (16, 64, 128):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        cfg = replace(TrainConfig(), batch_size=batch_size)
        seed_all(cfg.seed)
        model, optimizer = create(cfg, device)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        if device.type == "cuda":
            allocated_after_create = torch.cuda.memory_allocated(device)
        else:
            allocated_after_create = None
        indices = np.arange(batch_size)
        ids, labels, real = batch(train, indices, device)
        real_targets = int((labels != -100).sum().item())
        total_steps = math.ceil(len(train["input_ids"]) / cfg.batch_size) * cfg.epochs
        for step in range(warmup):
            one_update(model, optimizer, ids, labels, real, device, cfg, step + 1, total_steps)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for step in range(measured):
            one_update(
                model, optimizer, ids, labels, real, device, cfg, warmup + step + 1,
                total_steps,
            )
        sync(device)
        elapsed = time.perf_counter() - start
        gradient_bytes = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters() if p.grad is not None
        )
        optimizer_bytes = tensor_bytes(optimizer.state_dict()["state"])
        case = {
            "batch_blocks": batch_size,
            "positions_per_batch": batch_size * cfg.context,
            "real_targets_per_batch": real_targets,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "adam_state_bytes_including_scalar_steps": optimizer_bytes,
            "allocated_after_model_creation_bytes": allocated_after_create,
            "steady_allocated_bytes": (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else None
            ),
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
            "warmup_steps": warmup,
            "measured_steps": measured,
            "measured_elapsed_seconds": elapsed,
            "milliseconds_per_step": 1000 * elapsed / measured,
            "real_targets_per_second": real_targets * measured / elapsed,
            "padded_positions_per_second": batch_size * cfg.context * measured / elapsed,
        }
        cases.append(case)
        print(
            f"batch={batch_size} step_ms={case['milliseconds_per_step']:.2f} "
            f"target/s={case['real_targets_per_second']:.0f} "
            f"peak_allocated_MiB={case['peak_allocated_bytes'] / 2**20:.1f}"
            if device.type == "cuda" else
            f"batch={batch_size} step_ms={case['milliseconds_per_step']:.2f}"
        )
        del model, optimizer, ids, labels, real
    write_json(out, {
        "scope": "C18 train only; same first blocks reused for timing; no validation/test",
        "source_manifest_sha256": manifest_sha,
        "train_arrays_sha256": sha256(DATA / "train_blocks.npz"),
        "config_except_batch": asdict(TrainConfig()),
        "environment": environment(device),
        "timer": "warmup then one whole measured block, CUDA synchronized before and after",
        "memory_measure": "PyTorch allocator allocated/reserved; peak after warmup includes live state plus transient tensors; not total device use",
        "cases": cases,
    })


def depth(out: Path, progress: Path, seeds: tuple[int, ...]) -> None:
    if out.exists() or progress.exists():
        raise FileExistsError("previous depth output exists; choose another --out")
    manifest, manifest_sha = load_manifest()
    train = load_split(manifest, "train")
    validation = load_split(manifest, "validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in seeds:
        for layers in (2, 4, 8):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            cfg = replace(TrainConfig(), seed=seed, layers=layers)
            seed_all(seed)
            model, optimizer = create(cfg, device)
            order = np.random.default_rng(seed).permutation(len(train["input_ids"]))
            batches_per_epoch = math.ceil(len(order) / cfg.batch_size)
            plan_steps = batches_per_epoch * cfg.epochs
            total_loss, total_targets = 0.0, 0
            sync(device)
            start = time.perf_counter()
            for index in range(batches_per_epoch):
                indexes = order[index * cfg.batch_size:(index + 1) * cfg.batch_size]
                ids, labels, real = batch(train, indexes, device)
                summed, count = one_update(
                    model, optimizer, ids, labels, real, device, cfg, index + 1,
                    plan_steps,
                )
                total_loss += summed
                total_targets += count
            sync(device)
            train_seconds = time.perf_counter() - start
            train_peak = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
            diagnostic = evaluate_loss(
                model, train, fixed_train_indices(len(order), cfg), device, cfg
            )
            val = evaluate_loss(
                model, validation, np.arange(len(validation["input_ids"])), device, cfg
            )
            row = {
                "seed": seed,
                "layers": layers,
                "parameters": sum(p.numel() for p in model.parameters()),
                "first_epoch_updates": batches_per_epoch,
                "first_epoch_seen_real_targets": total_targets,
                "planned_total_epochs_for_lr": cfg.epochs,
                "first_epoch_online_nll": total_loss / total_targets,
                "first_epoch_fixed_train_nll": diagnostic["nll"],
                "first_epoch_validation_nll": val["nll"],
                "first_epoch_validation_targets": val["targets"],
                "train_seconds_synchronized": train_seconds,
                "training_real_targets_per_second": total_targets / train_seconds,
                "peak_training_allocated_bytes": train_peak,
            }
            rows.append(row)
            print(
                f"seed={seed} layers={layers} params={row['parameters']} "
                f"val_nll={val['nll']:.4f} time={train_seconds:.1f}s"
            )
            write_json(progress, {"completed_rows": rows, "expected_rows": len(seeds) * 3})
            del model, optimizer, ids, labels, real
    write_json(out, {
        "scope": "six fresh from-random first-epoch trains on C18 train; full C18 validation; test never opened",
        "source_manifest_sha256": manifest_sha,
        "train_arrays_sha256": sha256(DATA / "train_blocks.npz"),
        "validation_arrays_sha256": sha256(DATA / "validation_blocks.npz"),
        "fixed_config_except_layers_and_seed": asdict(TrainConfig()),
        "comparison": (
            "same BPE, context, batch size, full training block set, per-seed order, "
            "and first 334 steps of a planned 1336-step learning-rate curve; "
            "compute and random model shape differ"
        ),
        "environment": environment(device),
        "rows": rows,
    })
    progress.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["profile", "depth"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=12)
    parser.add_argument("--seeds", default="20260924,20260925")
    args = parser.parse_args()
    if args.warmup < 1 or args.measured < 1:
        raise ValueError("warmup and measured must be positive")
    out = args.out or (
        WORK / "results" / (
            "c20_resource_profile.json" if args.action == "profile" else
            "c20_depth_comparison.json"
        )
    )
    out = out if out.is_absolute() else WORK.parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "profile":
        profile(out, args.warmup, args.measured)
    else:
        seeds = tuple(int(piece) for piece in args.seeds.split(","))
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be a nonempty set of distinct integers")
        depth(out, out.with_name(out.stem + "_progress.json"), seeds)


if __name__ == "__main__":
    main()
