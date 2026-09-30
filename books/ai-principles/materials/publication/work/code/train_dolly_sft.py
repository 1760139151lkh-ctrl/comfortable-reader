"""C25: continue the actual C19 WikiText LM on response-only Dolly targets.

Modes: probe, train-full, train-lora. All use the same saved C19 weights,
existing tokenizer, and frozen sft_v2 split. No test split is opened here.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from prepare_dolly_sft import RAW, TOKENIZER_FILE, prompt_text, sha256
from train_causal_wikitext import (
    CausalLM,
    TrainConfig,
    autocast_for,
    batch as wiki_batch,
    evaluate_loss as evaluate_wiki_loss,
    generate,
    load_manifest as load_wiki_manifest,
    load_split as load_wiki_split,
)


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
DATA_DIR = WORK / "data/dolly15k/sft_v2"
DATA_MANIFEST = DATA_DIR / "manifest.json"
BASE_CHECKPOINT = WORK / "results/causal_lm_best.pt"
BASE_SUMMARY = WORK / "results/causal_lm_train.json"
CATEGORIES = (
    "brainstorming", "classification", "closed_qa", "creative_writing",
    "general_qa", "information_extraction", "open_qa", "summarization",
)
SEED = 20260924


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sft_manifest() -> tuple[dict, str]:
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    if file_sha(RAW) != manifest["raw_sha256"]:
        raise RuntimeError("raw Dolly source changed")
    if file_sha(TOKENIZER_FILE) != manifest["tokenizer_sha256"]:
        raise RuntimeError("C18 tokenizer changed")
    if manifest["context_positions"] != 128 or manifest["special_ids"] != {
        "pad": 0, "bos": 1, "eos": 2
    }:
        raise RuntimeError("SFT input format changed")
    if manifest["author_inspected_source_rows_excluded"] != [0, 1, 2]:
        raise RuntimeError("author-inspected source rows are no longer excluded")
    return manifest, file_sha(DATA_MANIFEST)


def load_sft_split(manifest: dict, name: str) -> dict[str, np.ndarray]:
    spec = manifest["splits"][name]
    path = ROOT / spec["arrays_path"]
    if file_sha(path) != spec["arrays_sha256"]:
        raise RuntimeError(f"SFT {name} arrays changed")
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    shape = (spec["records"], 128)
    if any(arrays[key].shape != shape for key in ("input_ids", "labels", "attention_mask")):
        raise RuntimeError(f"SFT {name} shape changed")
    if int((arrays["labels"] != -100).sum()) != spec["response_target_tokens_including_eos"]:
        raise RuntimeError(f"SFT {name} scored target count changed")
    if int(arrays["attention_mask"].sum()) != spec["real_input_positions"]:
        raise RuntimeError(f"SFT {name} real input count changed")
    if np.any((arrays["labels"] != -100) & ~arrays["attention_mask"]):
        raise RuntimeError("a scored label sits under a PAD input")
    if np.any(arrays["source_row"] < 3):
        raise RuntimeError("author-inspected source row leaked into SFT arrays")
    if not np.all(arrays["attention_mask"][:, 0]):
        raise RuntimeError("every example must start with real BOS")
    return arrays


def load_base() -> tuple[CausalLM, TrainConfig, dict]:
    summary = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    actual_sha = file_sha(BASE_CHECKPOINT)
    if actual_sha != summary["best_checkpoint_sha256"]:
        raise RuntimeError("C19 best checkpoint no longer matches its training record")
    wiki_manifest, wiki_sha = load_wiki_manifest()
    if summary["c18_manifest_sha256"] != wiki_sha:
        raise RuntimeError("C19 and C18 data identities no longer match")
    state = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = TrainConfig()
    if state["config"] != asdict(cfg) or state["c18_manifest_sha256"] != wiki_sha:
        raise RuntimeError("C19 checkpoint has different structure or source data")
    model = CausalLM(cfg)
    model.load_state_dict(state["model"])
    return model, cfg, {
        "c19_checkpoint_sha256": actual_sha,
        "c18_manifest_sha256": wiki_sha,
        "c19_selected_epoch": int(state["best_epoch"]),
        "c19_validation_nll": float(state["best_validation_nll"]),
        "wiki_manifest": wiki_manifest,
    }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.scale * self.b(self.a(value))


def attach_lora(model: CausalLM) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    routes = []
    for index, block in enumerate(model.blocks):
        for name in ("q", "v"):
            linear = getattr(block.read, name)
            setattr(block.read, name, LoRALinear(linear))
            routes.append(f"blocks.{index}.read.{name}")
        for part in (0, 2):
            block.ffn[part] = LoRALinear(block.ffn[part])
            routes.append(f"blocks.{index}.ffn.{part}")
    model.output = LoRALinear(model.output)
    routes.append("output")
    return routes


def parameter_counts(model: CausalLM) -> dict:
    return {
        "total_model_parameters_including_frozen": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_batch(part: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    ids = torch.as_tensor(part["input_ids"][indices].astype(np.int64), device=device)
    labels = torch.as_tensor(part["labels"][indices].astype(np.int64), device=device)
    real = torch.as_tensor(part["attention_mask"][indices], device=device)
    return ids, labels, real


@torch.no_grad()
def evaluate_sft(
    model: CausalLM, part: dict[str, np.ndarray], device: torch.device,
    batch_size: int = 64, with_categories: bool = False,
) -> dict:
    model.eval()
    total_loss, total_targets = 0.0, 0
    by_category = {
        name: {"loss_sum": 0.0, "targets": 0, "records": 0}
        for name in CATEGORIES
    }
    for offset in range(0, len(part["input_ids"]), batch_size):
        indices = np.arange(offset, min(offset + batch_size, len(part["input_ids"])))
        ids, labels, real = torch_batch(part, indices, device)
        with autocast_for(device):
            logits = model(ids, real)
        losses = F.cross_entropy(
            logits.float().transpose(1, 2), labels,
            ignore_index=-100, reduction="none",
        )
        mask = labels != -100
        row_loss = losses.sum(dim=1).detach().cpu().numpy()
        row_targets = mask.sum(dim=1).detach().cpu().numpy()
        total_loss += float(row_loss.sum())
        total_targets += int(row_targets.sum())
        if with_categories:
            for local, source_index in enumerate(indices):
                cat = CATEGORIES[int(part["category_id"][source_index])]
                by_category[cat]["loss_sum"] += float(row_loss[local])
                by_category[cat]["targets"] += int(row_targets[local])
                by_category[cat]["records"] += 1
    result = {
        "response_nll": total_loss / total_targets,
        "response_targets_including_eos": total_targets,
        "records": len(part["input_ids"]),
    }
    if with_categories:
        result["categories"] = {
            cat: {
                "records": item["records"],
                "targets": item["targets"],
                "response_nll": (
                    item["loss_sum"] / item["targets"] if item["targets"] else None
                ),
            }
            for cat, item in by_category.items()
        }
    return result


@torch.no_grad()
def evaluate_wiki(model: CausalLM, wiki_manifest: dict, cfg: TrainConfig, device: torch.device) -> dict:
    part = load_wiki_split(wiki_manifest, "validation")
    return evaluate_wiki_loss(model, part, np.arange(len(part["input_ids"])), device, cfg)


def fixed_validation_prompts(part: dict[str, np.ndarray]) -> list[dict]:
    raw_rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()]
    chosen = []
    used_categories = set()
    for source_row, category_id in zip(part["source_row"], part["category_id"]):
        cat = CATEGORIES[int(category_id)]
        if cat in used_categories:
            continue
        row = raw_rows[int(source_row)]
        chosen.append({
            "source_row": int(source_row),
            "category": cat,
            "prompt": prompt_text(row),
            "reference_response": row["response"].strip(),
        })
        used_categories.add(cat)
        if len(chosen) >= 4:
            break
    return chosen


def generate_fixed(
    model: CausalLM, tokenizer: Tokenizer, prompts: list[dict],
    device: torch.device, cfg: TrainConfig,
) -> list[dict]:
    samples = []
    for item in prompts:
        output = generate(
            model, tokenizer, item["prompt"], device, cfg,
            mode="greedy", seed=SEED, max_new=48,
        )
        samples.append({
            "source_row": item["source_row"],
            "category": item["category"],
            "prompt": item["prompt"],
            "reference_response": item["reference_response"],
            "continuation": output["continuation"],
            "new_token_count": output["new_token_count"],
            "ended_with_eos": output["ended_with_eos"],
        })
    return samples


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def probe(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    manifest, manifest_sha = load_sft_manifest()
    train = load_sft_split(manifest, "train")
    model, cfg, base = load_base()
    device = choose_device()
    model = model.to(device)
    index = np.asarray([0])
    ids, labels, real = torch_batch(train, index, device)
    if int(real.sum()) <= int((labels != -100).sum()):
        raise AssertionError("prompt visible positions must exceed response-only targets")
    first_target = int((labels[0] != -100).nonzero()[0].item())
    expected_boundary = int(train["prompt_tokens"][0])
    if first_target != expected_boundary:
        raise AssertionError("prompt/response boundary does not align")
    model.train()
    model.zero_grad(set_to_none=True)
    with autocast_for(device):
        logits = model(ids, real)
    active = int((labels != -100).sum().item())
    loss = F.cross_entropy(
        logits.float().transpose(1, 2), labels,
        ignore_index=-100, reduction="sum",
    ) / active
    loss.backward()
    grad_norm = math.sqrt(sum(
        float((p.grad.detach().float() ** 2).sum())
        for p in model.parameters() if p.grad is not None
    ))
    result = {
        "scope": "one current C19 checkpoint and one real Dolly training row; no optimizer step or checkpoint modification",
        "base": {k: v for k, v in base.items() if k != "wiki_manifest"},
        "sft_manifest_sha256": manifest_sha,
        "device": str(device),
        "source_row": int(train["source_row"][0]),
        "category": CATEGORIES[int(train["category_id"][0])],
        "prompt_tokens": expected_boundary,
        "response_tokens": int(train["response_tokens"][0]),
        "visible_input_positions": int(real.sum().item()),
        "scored_response_plus_eos_targets": active,
        "first_scored_input_position": first_target,
        "first_scored_target_id": int(labels[0, first_target]),
        "first_scored_target_matches_response": True,
        "response_only_nll_before_step": float(loss.detach()),
        "gradient_global_norm": grad_norm,
        "optimizer_step_executed": False,
    }
    write_json(out, result)
    print("probe source row", result["source_row"], "visible", result["visible_input_positions"],
          "scored", active, "first scored position", first_target)
    print("NLL", result["response_only_nll_before_step"], "grad norm", grad_norm)


def train(mode: str, out_dir: Path, epochs: int) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"keep old run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha = load_sft_manifest()
    train_part = load_sft_split(manifest, "train")
    val_part = load_sft_split(manifest, "validation")
    model, cfg, base = load_base()
    wiki_train_part = (
        load_wiki_split(base["wiki_manifest"], "train")
        if mode == "full-mix" else None
    )
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    lora_routes = attach_lora(model) if mode == "lora" else []
    counts = parameter_counts(model)
    device = choose_device()
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rate = 0.001 if mode == "lora" else 0.0002
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=rate, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.01 if mode in ("full", "full-mix") else 0.0,
    )
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    fixed_prompts = fixed_validation_prompts(val_part)
    base_sft = evaluate_sft(model, val_part, device, with_categories=True)
    base_wiki = evaluate_wiki(model, base["wiki_manifest"], cfg, device)
    base_generations = generate_fixed(model, tokenizer, fixed_prompts, device, cfg)
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    best_optimizer = None
    rng = np.random.default_rng(SEED)
    wiki_rng = np.random.default_rng(SEED + 9000)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_part["input_ids"]))
        running_loss, running_targets = 0.0, 0
        mixed_wiki_targets = 0
        mixed_wiki_loss_sum = 0.0
        steps = 0
        for offset in range(0, len(order), 64):
            indices = order[offset : offset + 64]
            ids, labels, real = torch_batch(train_part, indices, device)
            with autocast_for(device):
                logits = model(ids, real)
            total_loss = F.cross_entropy(
                logits.float().transpose(1, 2), labels,
                ignore_index=-100, reduction="sum",
            )
            count = int((labels != -100).sum().item())
            sft_loss = total_loss / count
            if wiki_train_part is not None:
                wiki_indices = wiki_rng.choice(
                    len(wiki_train_part["input_ids"]), size=64, replace=False
                )
                wiki_ids, wiki_labels, wiki_real = wiki_batch(
                    wiki_train_part, wiki_indices, device
                )
                with autocast_for(device):
                    wiki_logits = model(wiki_ids, wiki_real)
                wiki_total = F.cross_entropy(
                    wiki_logits.float().transpose(1, 2), wiki_labels,
                    ignore_index=-100, reduction="sum",
                )
                wiki_count = int((wiki_labels != -100).sum().item())
                loss = 0.9 * sft_loss + 0.1 * (wiki_total / wiki_count)
                mixed_wiki_targets += wiki_count
                mixed_wiki_loss_sum += float(wiki_total.detach())
            else:
                loss = sft_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite response loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            running_loss += float(total_loss.detach())
            running_targets += count
            steps += 1
        validation = evaluate_sft(model, val_part, device, with_categories=True)
        wiki = evaluate_wiki(model, base["wiki_manifest"], cfg, device)
        epoch_record = {
            "epoch": epoch,
            "optimizer_steps": steps,
            "online_train_response_nll": running_loss / running_targets,
            "mixed_wiki_online_nll": (
                mixed_wiki_loss_sum / mixed_wiki_targets
                if mixed_wiki_targets else None
            ),
            "mixed_wiki_targets_seen": mixed_wiki_targets,
            "validation": validation,
            "wiki_validation": wiki,
        }
        history.append(epoch_record)
        if validation["response_nll"] < best_val:
            best_val = validation["response_nll"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        print(
            mode, "epoch", epoch, "online", round(epoch_record["online_train_response_nll"], 4),
            "val", round(validation["response_nll"], 4),
            "old-wiki", round(wiki["nll"], 4),
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("no best checkpoint selected")
    model.load_state_dict(best_state)
    best_generations = generate_fixed(model, tokenizer, fixed_prompts, device, cfg)
    checkpoint = {
        "mode": mode,
        "base_checkpoint_sha256": base["c19_checkpoint_sha256"],
        "sft_manifest_sha256": manifest_sha,
        "selected_epoch": best_epoch,
        "validation_response_nll": best_val,
        "full_train_config": asdict(cfg),
        "adapter_routes": lora_routes,
        "lora_rank": 4 if mode == "lora" else None,
        "lora_alpha": 8.0 if mode == "lora" else None,
        "model": (
            {name: p.detach().cpu().clone()
             for name, p in model.named_parameters() if p.requires_grad}
            if mode == "lora"
            else {name: tensor.detach().cpu().clone()
                  for name, tensor in model.state_dict().items()}
        ),
        "optimizer": best_optimizer,
    }
    checkpoint_path = out_dir / "best.pt"
    temp_checkpoint = checkpoint_path.with_name("best.pt.tmp")
    torch.save(checkpoint, temp_checkpoint)
    os.replace(temp_checkpoint, checkpoint_path)
    summary = {
        "scope": "C19 exact 4.93M random-pretrained WikiText LM continued on Dolly sft_v2 response-only labels; no test opened, no external pretrained weights",
        "mode": mode,
        "base": {k: v for k, v in base.items() if k != "wiki_manifest"},
        "sft_manifest_sha256": manifest_sha,
        "data_split": {
            "train_records": manifest["splits"]["train"]["records"],
            "validation_records": manifest["splits"]["validation"]["records"],
            "test_records_not_opened": manifest["splits"]["test"]["records"],
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "parameters": counts,
        "adapter_routes": lora_routes,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": rate,
            "weight_decay": 0.01 if mode in ("full", "full-mix") else 0.0,
            "batch_records": 64,
            "epochs_planned": epochs,
            "gradient_clip_norm": 1.0,
            "mixed_old_wikitext": (
                "each Dolly batch also draws 64 fixed-split C18 train blocks; "
                "loss=0.9*mean_response_NLL+0.1*mean_WikiText_target_NLL"
                if mode == "full-mix" else None
            ),
        },
        "baseline_validation": base_sft,
        "baseline_wiki_validation": base_wiki,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_response_nll": best_val,
        "best_wiki_validation_nll": history[best_epoch - 1]["wiki_validation"]["nll"],
        "fixed_validation_prompts": fixed_prompts,
        "baseline_generations": base_generations,
        "selected_generations": best_generations,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": file_sha(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "test_opened": False,
    }
    write_json(out_dir / "train.json", summary)
    print(mode, "best epoch", best_epoch, "val", best_val, "checkpoint", summary["checkpoint_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "train-full", "train-lora", "train-full-mix"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.mode == "probe":
        out = args.out or Path("work/results/c25_sft_step_probe.json")
        path = out if out.is_absolute() else ROOT / out
        probe(path)
    else:
        default = {
            "train-full": "work/runs/c25_full",
            "train-lora": "work/runs/c25_lora",
            "train-full-mix": "work/runs/c25_full_mix",
        }[args.mode]
        out = args.out or Path(default)
        path = out if out.is_absolute() else ROOT / out
        chosen_mode = {
            "train-full": "full",
            "train-lora": "lora",
            "train-full-mix": "full-mix",
        }[args.mode]
        train(chosen_mode, path, args.epochs)


if __name__ == "__main__":
    main()
