"""C38: learn assistant call/answer tokens from C25's own small language model.

The trainer opens only the train and validation JSONL files. Test is not read
or scored here. A separate evaluator must open it after checkpoint choice.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from c38_tool_data import tokenized
from c38_tool_data_v3 import tokenized_v3
from train_causal_wikitext import CausalLM, TrainConfig

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
DATA = ROOT / "data/c38_tool_trajectories_v1"
PARENT = ROOT / "runs/c25_full_mix_trial/best.pt"
TOKENIZER = ROOT / "data/wikitext2_causal/tokenizer.json"
KINDS = ("multiply", "title", "direct", "missing")
SEED = 20260925


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["parent_c25_full_mix_checkpoint_sha256"] != sha(PARENT) or manifest["tokenizer_sha256"] != sha(TOKENIZER):
        raise RuntimeError("C38 trajectory source and C25/C18 model identity disagree")
    for name in ("train.jsonl", "validation.jsonl"):
        if manifest["files"][name] != sha(data_dir / name):
            raise RuntimeError(name + " changed")
    tok = Tokenizer.from_file(str(TOKENIZER))
    return manifest, tok


def load_rows(name, tok, data_dir: Path, format_version: int):
    rows = [json.loads(line) for line in (data_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    encode_row = tokenized_v3 if format_version == 3 else tokenized
    arrays = [encode_row(tok, row) for row in rows]
    inp = np.array([row["input_ids"] for row in arrays], dtype=np.int64)
    labels = np.array([row["labels"] for row in arrays], dtype=np.int64)
    real = np.array([row["attention_mask"] for row in arrays], dtype=np.bool_)
    final_mask = np.array([row["assistant_final_target_mask"] for row in arrays], dtype=np.bool_)
    if inp.shape != labels.shape or inp.shape != real.shape or inp.shape[1] != 128:
        raise RuntimeError("trajectory tensor dimensions wrong")
    if np.any((labels != -100) & ~real):
        raise RuntimeError("padding target was scored")
    assert np.all(~final_mask | (labels != -100))
    return rows, arrays, inp, labels, real, final_mask


def parent_model(device):
    cfg = TrainConfig()
    state = torch.load(PARENT, map_location="cpu", weights_only=True)
    if state["mode"] != "full-mix" or state["full_train_config"] != asdict(cfg) or state["selected_epoch"] != 3:
        raise RuntimeError("unexpected C25 parent model")
    model = CausalLM(cfg).to(device)
    model.load_state_dict(state["model"])
    return model, cfg, state


def autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def loss_matrix(model, inp, labels, real, device):
    x = torch.as_tensor(inp, dtype=torch.long, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    mask = torch.as_tensor(real, dtype=torch.bool, device=device)
    with autocast(device):
        logits = model(x, mask)
    per_target = F.cross_entropy(logits.transpose(1, 2).float(), y, ignore_index=-100, reduction="none")
    return per_target, y


@torch.inference_mode()
def evaluate(model, data, device):
    rows, _, inp, labels, real, _ = data
    model.eval()
    totals = {kind: {"loss": 0.0, "targets": 0, "rows": 0} for kind in KINDS}
    for start in range(0, len(rows), 32):
        end = start + 32
        values, y = loss_matrix(model, inp[start:end], labels[start:end], real[start:end], device)
        row_sums = values.sum(dim=1).cpu().numpy()
        counts = (y != -100).sum(dim=1).cpu().numpy()
        for i, row in enumerate(rows[start:end]):
            item = totals[row["kind"]]
            item["loss"] += float(row_sums[i])
            item["targets"] += int(counts[i])
            item["rows"] += 1
    all_loss = sum(item["loss"] for item in totals.values())
    all_targets = sum(item["targets"] for item in totals.values())
    by_kind = {name: {"nll": item["loss"] / item["targets"], "targets": item["targets"], "rows": item["rows"]} for name, item in totals.items()}
    return {"assistant_target_nll": all_loss / all_targets, "assistant_targets": all_targets,
            "macro_kind_nll": sum(item["nll"] for item in by_kind.values()) / len(by_kind),
            "by_kind": by_kind}


def train(out_dir: Path, steps: int, lr: float, validate_every: int, per_kind: int, data_dir: Path, final_weight: float):
    if out_dir.exists():
        raise RuntimeError("output run exists; use a fresh directory")
    if not (steps > 0 and validate_every > 0 and per_kind > 0 and 0 < lr < 0.01 and 1 <= final_weight <= 10):
        raise ValueError("invalid training plan")
    manifest, tok = source(data_dir)
    format_version = manifest.get("format_version", 1)
    if format_version not in (1, 3):
        raise RuntimeError("unknown C38 trajectory token-segment version")
    train_data = load_rows("train", tok, data_dir, format_version)
    val_data = load_rows("validation", tok, data_dir, format_version)
    rows, arrays, inp, labels, real, final_mask = train_data
    assert len(rows) == manifest["stats"]["train"]["rows"]
    assert len(val_data[0]) == manifest["stats"]["validation"]["rows"]
    pools = {kind: np.array([i for i, row in enumerate(rows) if row["kind"] == kind]) for kind in KINDS}
    assert all(len(v) > 0 for v in pools.values())
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, cfg, parent = parent_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    out_dir.mkdir(parents=True)
    first_before = model.output.weight.detach().float().clone()
    initial = evaluate(model, val_data, device)
    best = initial["macro_kind_nll"]
    best_step = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"step": 0, "validation": initial}]
    started = time.perf_counter()
    first_batch = None
    for step in range(1, steps + 1):
        model.train()
        indices = np.concatenate([rng.choice(pools[kind], size=per_kind, replace=True) for kind in KINDS])
        rng.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        per_target, target = loss_matrix(model, inp[indices], labels[indices], real[indices], device)
        count = (target != -100).sum()
        weights = 1 + (final_weight - 1) * torch.as_tensor(final_mask[indices], device=device, dtype=torch.float32)
        weighted_count = (weights * (target != -100)).sum()
        loss = (per_target * weights).sum() / weighted_count
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step == 1:
            first_batch = {
                "assistant_targets": int(count), "weighted_assistant_target_total": float(weighted_count), "loss_before_update": float(loss.detach()),
                "gradient_norm_before_clip": grad_norm,
                "output_weight_change_l2": float((model.output.weight.detach().float() - first_before).norm()),
                "kind_counts": {kind: sum(rows[int(i)]["kind"] == kind for i in indices) for kind in KINDS},
                "one_real_trace": {"row": rows[int(indices[0])], "source_counts": {k: arrays[int(indices[0])][k] for k in ("assistant_first_targets", "observation_visible_unscored_tokens", "assistant_final_targets")}},
            }
            del first_before
        if step % validate_every == 0 or step == steps:
            measure = evaluate(model, val_data, device)
            record = {"step": step, "train_last_batch_nll": float(loss.detach()), "validation": measure}
            history.append(record)
            print(json.dumps({"step": step, "train_nll": record["train_last_batch_nll"], "validation_macro_nll": measure["macro_kind_nll"], "validation_target_weighted_nll": measure["assistant_target_nll"], "by_kind": {k: round(v["nll"], 4) for k, v in measure["by_kind"].items()}}, ensure_ascii=False), flush=True)
            if measure["macro_kind_nll"] < best:
                best = measure["macro_kind_nll"]
                best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    checkpoint = out_dir / "best.pt"
    torch.save({"model": best_state, "config": asdict(cfg), "parent_checkpoint_sha256": sha(PARENT),
                "data_manifest_sha256": sha(data_dir / "manifest.json"), "selected_step": best_step,
                "validation_macro_kind_nll": best, "objective": "assistant action and final answer targets only"}, checkpoint)
    report = {
        "task": "C38 narrow real model tool-action SFT, no external strong weights, test unopened",
        "command": f"python work/code/c38_tool_train.py --out-dir {out_dir.relative_to(PACKAGE)} --steps {steps} --lr {lr} --validate-every {validate_every} --per-kind {per_kind} --data-dir {data_dir.relative_to(PACKAGE)} --final-weight {final_weight}",
        "parent_c25_checkpoint_sha256": sha(PARENT), "data_manifest_sha256": sha(data_dir / "manifest.json"),
        "data_dir": str(data_dir.relative_to(PACKAGE)).replace("\\", "/"), "format_version": format_version,
        "tokenizer_sha256": sha(TOKENIZER), "seed": SEED, "device": str(device), "torch": torch.__version__,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "training_rows": len(rows), "validation_rows": len(val_data[0]),
        "test_file_loaded_during_train": False,
        "plan": {"steps": steps, "lr": lr, "validate_every": validate_every, "per_kind": per_kind, "batch": per_kind * len(KINDS), "optimizer": "AdamW", "loss": "mean over assistant-target IDs with optional extra final-answer token weight; prompt/tool observation targets masked", "final_weight": final_weight, "checkpoint_selection": "equal-weight macro mean of four validation kind NLLs, so plentiful multiply targets do not alone choose the model"},
        "first_batch": first_batch, "history": history, "selected_step": best_step,
        "selected_validation_macro_kind_nll": best,
        "checkpoint": str(checkpoint.relative_to(PACKAGE)).replace("\\", "/"), "checkpoint_sha256": sha(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
    }
    (out_dir / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_step": best_step, "best_validation_macro_kind_nll": best, "checkpoint_sha256": report["checkpoint_sha256"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--per-kind", type=int, default=6)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--final-weight", type=float, default=1.0)
    args = parser.parse_args()
    destination = args.out_dir.resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("output directory must stay in the current work directory")
    data_dir = args.data_dir.resolve()
    if not data_dir.is_relative_to(ROOT / "data"):
        raise ValueError("data directory must stay in the current work/data directory")
    train(destination, args.steps, args.lr, args.validate_every, args.per_kind, data_dir, args.final_weight)
