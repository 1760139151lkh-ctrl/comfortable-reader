"""C19: train a small causal Transformer from random weights on C18 WikiText blocks.

Examples, from the project root:
    python work/code/train_causal_wikitext.py probe
    python work/code/train_causal_wikitext.py train --epochs 4 --stop-after-epoch 1
    python work/code/train_causal_wikitext.py train --epochs 4 --resume
    python work/code/train_causal_wikitext.py evaluate

The test arrays are opened only by evaluate, after best-checkpoint selection.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from transformer_bridge import MultiHeadRead, positional_encoding


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data" / "wikitext2_causal"
RESULTS = WORK / "results"
MANIFEST_PATH = DATA / "manifest.json"
LATEST = RESULTS / "causal_lm_latest.pt"
BEST = RESULTS / "causal_lm_best.pt"
HISTORY = RESULTS / "causal_lm_history.jsonl"
SUMMARY = RESULTS / "causal_lm_train.json"
EVALUATION = RESULTS / "causal_lm_evaluation.json"
PROBE = RESULTS / "causal_lm_step_probe.json"
SEED = 20260924
PAD, BOS, EOS = 0, 1, 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def select_output_dir(path_text: str) -> None:
    """Keep a reader's new run separate from the published reference run."""
    global RESULTS, LATEST, BEST, HISTORY, SUMMARY, EVALUATION, PROBE
    chosen = Path(path_text)
    RESULTS = chosen if chosen.is_absolute() else WORK.parent / chosen
    LATEST = RESULTS / "causal_lm_latest.pt"
    BEST = RESULTS / "causal_lm_best.pt"
    HISTORY = RESULTS / "causal_lm_history.jsonl"
    SUMMARY = RESULTS / "causal_lm_train.json"
    EVALUATION = RESULTS / "causal_lm_evaluation.json"
    PROBE = RESULTS / "causal_lm_step_probe.json"


def append_event(obj: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
        stream.flush()


@dataclass(frozen=True)
class TrainConfig:
    vocab: int = 8192
    context: int = 128
    width: int = 192
    heads: int = 4
    inner: int = 768
    layers: int = 4
    batch_size: int = 64
    epochs: int = 4
    warmup_steps: int = 100
    peak_lr: float = 0.0003
    final_lr: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    seed: int = SEED
    diagnostic_train_blocks: int = 1024
    checkpoint_interval_steps: int = 100


class CausalBlock(nn.Module):
    """C17's self-read + FFN order, with causal visibility supplied by the caller."""

    def __init__(self, width: int, heads: int, inner: int):
        super().__init__()
        self.read = MultiHeadRead(width, heads)
        self.norm_read = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, inner), nn.ReLU(), nn.Linear(inner, width))
        self.norm_ffn = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        read, _ = self.read(x, x, allowed)
        x = self.norm_read(x + read)
        return self.norm_ffn(x + self.ffn(x))


class CausalLM(nn.Module):
    """The C17 post-LN self-reader, now repeated four times with C18 PAD masking."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.width, padding_idx=PAD)
        self.blocks = nn.ModuleList(
            [CausalBlock(cfg.width, cfg.heads, cfg.inner) for _ in range(cfg.layers)]
        )
        self.output = nn.Linear(cfg.width, cfg.vocab)

    def forward(self, ids: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        if ids.shape != real.shape or ids.ndim != 2:
            raise ValueError("ids and real must both be [batch, positions]")
        if ids.shape[1] > self.cfg.context:
            raise ValueError("context overflow")
        if not bool(real[:, 0].all()):
            raise ValueError("each sequence must begin with a real token")
        length = ids.shape[1]
        triangle = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril()
        allowed = triangle[None, None, :, :] & real[:, None, None, :]
        x = self.embed(ids) * math.sqrt(self.cfg.width)
        x = x + positional_encoding(length, self.cfg.width, ids.device)[None]
        for block in self.blocks:
            x = block(x, allowed)
        return self.output(x)


def load_manifest() -> tuple[dict, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256(DATA / "tokenizer.json") != manifest["tokenizer"]["model_sha256"]:
        raise ValueError("C18 tokenizer no longer matches its manifest")
    ids = manifest["tokenizer"]["special_ids"]
    if ids != {"<|pad|>": PAD, "<|bos|>": BOS, "<|eos|>": EOS}:
        raise ValueError("unexpected special IDs")
    return manifest, sha256(MANIFEST_PATH)


def load_split(manifest: dict, name: str) -> dict[str, np.ndarray]:
    spec = manifest["splits"][name]
    path = WORK / spec["arrays_path"]
    if sha256(path) != spec["arrays_sha256"]:
        raise ValueError(f"C18 {name} arrays changed")
    with np.load(path) as npz:
        part = {key: npz[key].copy() for key in npz.files}
    if part["input_ids"].shape != (spec["blocks"], 128):
        raise ValueError(f"{name} shape changed")
    if not np.array_equal(part["attention_mask"], part["labels"] != -100):
        raise ValueError(f"{name} real-target mask changed")
    if int(part["attention_mask"].sum()) != spec["actual_targets_including_eos"]:
        raise ValueError(f"{name} target count changed")
    if not np.all(part["attention_mask"][:, 0]):
        raise ValueError("empty block")
    return part


def batch(part: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    ids = torch.as_tensor(part["input_ids"][indices].astype(np.int64), device=device)
    labels = torch.as_tensor(part["labels"][indices].astype(np.int64), device=device)
    real = torch.as_tensor(part["attention_mask"][indices], device=device)
    return ids, labels, real


def loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    # The softmax/loss is in float32 even when CUDA matrix products use BF16.
    total = F.cross_entropy(
        logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="sum"
    )
    count = int((labels != -100).sum().item())
    if count == 0:
        raise ValueError("empty target batch")
    return total, count


def autocast_for(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )


def learning_rate(step: int, total_steps: int, cfg: TrainConfig) -> float:
    if step <= cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    proportion = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.final_lr + 0.5 * (cfg.peak_lr - cfg.final_lr) * (1 + math.cos(math.pi * proportion))


def set_lr(optimizer: torch.optim.Optimizer, rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = rate


def fixed_train_indices(n: int, cfg: TrainConfig) -> np.ndarray:
    return np.random.default_rng(cfg.seed + 100000).choice(
        n, size=min(cfg.diagnostic_train_blocks, n), replace=False
    )


@torch.no_grad()
def evaluate_loss(
    model: CausalLM, part: dict[str, np.ndarray], indices: np.ndarray, device: torch.device,
    cfg: TrainConfig,
) -> dict:
    model.eval()
    total_loss, total_targets = 0.0, 0
    for offset in range(0, len(indices), cfg.batch_size):
        ids, labels, real = batch(part, indices[offset:offset + cfg.batch_size], device)
        with autocast_for(device):
            logits = model(ids, real)
        sum_loss, count = loss_sum(logits, labels)
        total_loss += float(sum_loss)
        total_targets += count
    nll = total_loss / total_targets
    return {"nll": nll, "perplexity": math.exp(nll), "targets": total_targets, "blocks": len(indices)}


def unigram_baseline(
    train_part: dict[str, np.ndarray], eval_part: dict[str, np.ndarray], vocab: int
) -> dict:
    train_labels = train_part["labels"]
    counts = np.bincount(train_labels[train_labels >= 0].ravel(), minlength=vocab).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + vocab)
    eval_labels = eval_part["labels"]
    targets = eval_labels[eval_labels >= 0].ravel()
    nll = float(-np.log(probabilities[targets]).mean())
    return {"nll": nll, "perplexity": math.exp(nll), "targets": int(len(targets)),
            "rule": "training target counts +1 Laplace smoothing; no context"}


def environment(device: torch.device) -> dict:
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "device": str(device)}
    if device.type == "cuda":
        result.update({"gpu": torch.cuda.get_device_name(device),
                       "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
                       "bf16_supported": torch.cuda.is_bf16_supported()})
    return result


def atomic_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def checkpoint_state(
    model: CausalLM, optimizer: torch.optim.Optimizer, cfg: TrainConfig, manifest_sha: str,
    epoch: int, next_batch: int, global_step: int, best_val: float, best_epoch: int,
    epoch_loss_sum: float, epoch_targets: int,
) -> dict:
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": asdict(cfg), "c18_manifest_sha256": manifest_sha,
        "epoch": epoch, "next_batch": next_batch, "global_step": global_step,
        "best_validation_nll": best_val, "best_epoch": best_epoch,
        "epoch_online_loss_sum": epoch_loss_sum, "epoch_online_targets": epoch_targets,
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def load_checkpoint(path: Path, device: torch.device, cfg: TrainConfig, manifest_sha: str) -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    if state["config"] != asdict(cfg) or state["c18_manifest_sha256"] != manifest_sha:
        raise ValueError("checkpoint belongs to a different model, schedule or C18 data")
    return state


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create(cfg: TrainConfig, device: torch.device):
    set_seed(cfg.seed)
    model = CausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps, weight_decay=cfg.weight_decay,
    )
    return model, optimizer


def probe() -> None:
    cfg = TrainConfig()
    manifest, manifest_sha = load_manifest()
    part = load_split(manifest, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA training protocol requires BF16 support")
    model, optimizer = create(cfg, device)
    ids, labels, real = batch(part, np.arange(2), device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with autocast_for(device):
        logits = model(ids, real)
    summed, count = loss_sum(logits, labels)
    loss = summed / count
    loss.backward()
    raw_norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg.clip_norm, error_if_nonfinite=True
    ))
    set_lr(optimizer, learning_rate(1, math.ceil(len(part["input_ids"]) / cfg.batch_size) * cfg.epochs, cfg))
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        model.eval()
        with autocast_for(device):
            after = model(ids, real)
        after_sum, _ = loss_sum(after, labels)
    report = {
        "scope": "first two actual train blocks; one update; disposable model, not full training",
        "c18_manifest_sha256": manifest_sha, "config": asdict(cfg),
        "parameters": sum(p.numel() for p in model.parameters()),
        "targets": count, "nll_before_step": float(loss.detach()),
        "nll_same_batch_after_step": float(after_sum / count),
        "raw_gradient_norm": raw_norm, "clip_norm": cfg.clip_norm,
        "first_step_lr": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": time.perf_counter() - start,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "environment": environment(device),
    }
    write_json(PROBE, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(cfg: TrainConfig, resume: bool, stop_after_epoch: int | None) -> None:
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= cfg.epochs:
        raise ValueError("stop-after-epoch must be between 1 and planned epochs")
    manifest, manifest_sha = load_manifest()
    train_part = load_split(manifest, "train")
    val_part = load_split(manifest, "validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA training protocol requires BF16 support")
    model, optimizer = create(cfg, device)
    n = len(train_part["input_ids"])
    batches_per_epoch = math.ceil(n / cfg.batch_size)
    total_steps = batches_per_epoch * cfg.epochs
    epoch, next_batch, global_step = 0, 0, 0
    best_val, best_epoch = float("inf"), 0
    epoch_loss_sum, epoch_targets = 0.0, 0
    if resume:
        state = load_checkpoint(LATEST, device, cfg, manifest_sha)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, next_batch, global_step = state["epoch"], state["next_batch"], state["global_step"]
        best_val, best_epoch = state["best_validation_nll"], state["best_epoch"]
        epoch_loss_sum, epoch_targets = state["epoch_online_loss_sum"], state["epoch_online_targets"]
        torch.set_rng_state(state["torch_cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda_rng_all"]])
        append_event({"event": "resume", "epoch_zero_based": epoch,
                      "next_batch_zero_based": next_batch, "global_step": global_step})
    else:
        if LATEST.exists() or BEST.exists() or HISTORY.exists():
            raise FileExistsError("C19 run already exists; use --resume or move old run files first")
        append_event({"event": "start", "seed": cfg.seed, "planned_epochs": cfg.epochs,
                      "batches_per_epoch": batches_per_epoch, "parameters":
                      sum(p.numel() for p in model.parameters()),
                      "c18_manifest_sha256": manifest_sha, "environment": environment(device)})
    if epoch >= cfg.epochs:
        raise ValueError("checkpoint has already completed all planned epochs")
    diag_indices = fixed_train_indices(n, cfg)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_completed_epoch = epoch
    for current_epoch in range(epoch, cfg.epochs):
        if stop_after_epoch is not None and current_epoch >= stop_after_epoch:
            break
        ordering = np.random.default_rng(cfg.seed + current_epoch).permutation(n)
        first_batch = next_batch if current_epoch == epoch else 0
        if current_epoch != epoch:
            epoch_loss_sum, epoch_targets = 0.0, 0
        model.train()
        for batch_number in range(first_batch, batches_per_epoch):
            selected = ordering[batch_number * cfg.batch_size:(batch_number + 1) * cfg.batch_size]
            ids, labels, real = batch(train_part, selected, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_for(device):
                logits = model(ids, real)
            summed, count = loss_sum(logits, labels)
            loss = summed / count
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {global_step + 1}")
            loss.backward()
            raw_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.clip_norm, error_if_nonfinite=True
            ))
            rate = learning_rate(global_step + 1, total_steps, cfg)
            set_lr(optimizer, rate)
            optimizer.step()
            global_step += 1
            epoch_loss_sum += float(summed.detach())
            epoch_targets += count
            if global_step == 1 or global_step % 50 == 0:
                event = {"event": "step", "epoch": current_epoch + 1,
                         "batch_one_based": batch_number + 1, "global_step": global_step,
                         "batch_nll_before_update": float(loss.detach()),
                         "lr": rate, "gradient_norm_before_clip": raw_norm,
                         "seen_real_targets_in_epoch": epoch_targets}
                append_event(event)
                print(json.dumps(event, ensure_ascii=False), flush=True)
            if global_step % cfg.checkpoint_interval_steps == 0:
                atomic_checkpoint(
                    LATEST, checkpoint_state(
                        model, optimizer, cfg, manifest_sha, current_epoch, batch_number + 1,
                        global_step, best_val, best_epoch, epoch_loss_sum, epoch_targets,
                    )
                )
        train_online = epoch_loss_sum / epoch_targets
        train_probe = evaluate_loss(model, train_part, diag_indices, device, cfg)
        validation = evaluate_loss(
            model, val_part, np.arange(len(val_part["input_ids"])), device, cfg
        )
        if not math.isfinite(validation["nll"]):
            raise FloatingPointError("nonfinite validation loss")
        improved = validation["nll"] < best_val
        if improved:
            best_val, best_epoch = validation["nll"], current_epoch + 1
        end_state = checkpoint_state(
            model, optimizer, cfg, manifest_sha, current_epoch + 1, 0, global_step,
            best_val, best_epoch, 0.0, 0,
        )
        if improved:
            atomic_checkpoint(BEST, end_state)
        atomic_checkpoint(LATEST, end_state)
        last_completed_epoch = current_epoch + 1
        event = {
            "event": "epoch_end", "epoch": current_epoch + 1, "global_step": global_step,
            "train_online_nll_at_changing_weights": train_online,
            "train_online_targets": epoch_targets,
            "fixed_train_probe": train_probe, "validation": validation,
            "best_epoch_so_far": best_epoch, "best_validation_nll": best_val,
            "best_checkpoint_updated": improved, "elapsed_seconds_this_invocation": time.perf_counter() - start,
        }
        append_event(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        next_batch = 0
    summary = {
        "scope": "full C18 train and validation blocks, from random weights; no test opened",
        "c18_manifest_sha256": manifest_sha, "config": asdict(cfg),
        "parameters": sum(p.numel() for p in model.parameters()),
        "completed_epochs": last_completed_epoch, "planned_epochs": cfg.epochs,
        "global_steps": global_step, "best_epoch": best_epoch,
        "best_validation_nll": best_val, "best_validation_perplexity": math.exp(best_val),
        "validation_unigram_baseline": unigram_baseline(train_part, val_part, cfg.vocab),
        "latest_checkpoint_sha256": sha256(LATEST),
        "best_checkpoint_sha256": sha256(BEST),
        "test_opened": False, "elapsed_seconds_this_invocation": time.perf_counter() - start,
        "peak_gpu_allocated_bytes_this_invocation":
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "environment": environment(device),
    }
    write_json(SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


@torch.no_grad()
def generate(
    model: CausalLM, tokenizer: Tokenizer, prompt: str, device: torch.device,
    cfg: TrainConfig, mode: str, seed: int, max_new: int = 100,
) -> dict:
    model.eval()
    original = [BOS] + tokenizer.encode(prompt, add_special_tokens=False).ids
    produced = original.copy()
    rng = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new):
        window = torch.as_tensor(produced[-cfg.context:], device=device, dtype=torch.long)[None]
        real = torch.ones_like(window, dtype=torch.bool)
        with autocast_for(device):
            logits = model(window, real)[0, -1].float()
        logits[[PAD, BOS]] = float("-inf")
        if mode == "greedy":
            next_id = int(logits.argmax())
        else:
            top_values, top_indices = logits.topk(40)
            probabilities = F.softmax(top_values / 0.8, dim=-1)
            chosen = torch.multinomial(probabilities, 1, generator=rng)
            next_id = int(top_indices[chosen])
        produced.append(next_id)
        if next_id == EOS:
            break
    continuation = produced[len(original):]
    visible = [item for item in continuation if item != EOS]
    return {
        "prompt": prompt, "mode": mode, "seed": seed if mode != "greedy" else None,
        "new_token_ids": continuation, "new_token_count": len(continuation),
        "ended_with_eos": bool(continuation and continuation[-1] == EOS),
        "continuation": tokenizer.decode(visible, skip_special_tokens=False),
    }


def evaluate(cfg: TrainConfig) -> None:
    manifest, manifest_sha = load_manifest()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary["completed_epochs"] != cfg.epochs:
        raise ValueError("the planned train run is incomplete")
    if summary["c18_manifest_sha256"] != manifest_sha:
        raise ValueError("C18 manifest changed after training")
    if EVALUATION.exists():
        raise FileExistsError("test already evaluated; preserve this result rather than retesting selections")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(BEST, device, cfg, manifest_sha)
    if sha256(BEST) != summary["best_checkpoint_sha256"]:
        raise ValueError("best checkpoint changed")
    model, _ = create(cfg, device)
    model.load_state_dict(state["model"])
    train_part = load_split(manifest, "train")
    test_part = load_split(manifest, "test")
    start = time.perf_counter()
    test_result = evaluate_loss(
        model, test_part, np.arange(len(test_part["input_ids"])), device, cfg
    )
    baseline = unigram_baseline(train_part, test_part, cfg.vocab)
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    # Fixed before seeing the test score or generated text.
    prompts = [" = History = \n", "The city of", "In the early"]
    generations = [
        generate(model, tokenizer, p, device, cfg, mode, cfg.seed + index)
        for index, p in enumerate(prompts)
        for mode in ("greedy", "sample_top40")
    ]
    result = {
        "scope": "one-time held-out test and free-running generation from validation-selected best checkpoint",
        "c18_manifest_sha256": manifest_sha, "checkpoint_sha256": sha256(BEST),
        "selected_epoch": state["best_epoch"], "selected_validation_nll": state["best_validation_nll"],
        "test": test_result, "unigram_baseline_test": baseline,
        "generation_protocol": {
            "prompts": prompts, "max_new_tokens": 100, "context_positions": cfg.context,
            "greedy": "argmax", "sample": "top_k=40, temperature=0.8, fixed seed",
            "forbidden_output_ids": [PAD, BOS], "eos_stops": True,
        },
        "generations": generations, "elapsed_seconds": time.perf_counter() - start,
        "environment": environment(device),
    }
    write_json(EVALUATION, result)
    print(json.dumps({k: v for k, v in result.items() if k != "generations"},
                     ensure_ascii=False, indent=2))
    for sample in generations:
        print(json.dumps({k: v for k, v in sample.items() if k != "new_token_ids"},
                         ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["probe", "train", "evaluate"])
    parser.add_argument("--output-dir", default="work/results",
                        help="keep a new reader run in its own directory")
    parser.add_argument("--epochs", type=int, default=4, help="total planned epochs, fixed across resume")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    select_output_dir(args.output_dir)
    cfg = TrainConfig(epochs=args.epochs)
    if args.action == "probe":
        probe()
    elif args.action == "train":
        train(cfg, args.resume, args.stop_after_epoch)
    else:
        evaluate(cfg)


if __name__ == "__main__":
    main()
