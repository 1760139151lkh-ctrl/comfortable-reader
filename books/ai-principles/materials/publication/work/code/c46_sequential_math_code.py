"""Continue C27's same math model on short code, with explicit retention routes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import arithmetic_curriculum as math_task
import code_expression_lab as code_task

ROOT = math_task.ROOT
PARENT = ROOT / "work/runs/c27_digit_addition_long_trial/best.pt"
EXPECTED_PARENT = "02be020308fabdc0b7ea05f800ac424d8c062e525130f91cae9aa30825464115"
IMPORTANCE_PATH = ROOT / "work/runs/c46_importance_first/importance.pt"
EXPECTED_IMPORTANCE = "a64947e0f24b02325c05331366dffa516a2649dd0130551117d319ee5f267766"
SEED = 4602
LR = 0.0002
LAMBDA = 100000.0


def mean_nll(model, records, make_batch, tokenizer, device):
    inputs, labels, real = make_batch(records, tokenizer, device)
    with math_task.autocast_for(device):
        logits = model(inputs, real)
    return F.cross_entropy(logits.float().transpose(1, 2), labels,
                           ignore_index=-100, reduction="sum") / (labels != -100).sum()


def validate(model, tokenizer, device, math_val, code_val) -> dict:
    math_nll = math_task.teacher_nll(model, math_val, tokenizer, device)
    code_nll = code_task.teacher_nll(model, code_val, tokenizer, device)
    math_free = math_task.free_generate(model, math_val, tokenizer, device)
    code_free = code_task.free_generate(model, code_val, tokenizer, device, code_task.VALIDATION_TEST_INPUTS)
    return {
        "old_math_validation_nll": math_nll,
        "new_code_validation_nll": code_nll,
        "old_math_validation_exact_and_eos": math_free["exact_answer_and_eos"],
        "old_math_validation_total": len(math_val),
        "new_code_validation_symbolically_equivalent": code_free["symbolically_equivalent"],
        "new_code_validation_total": len(code_val),
        "math_examples": math_free["examples"],
        "code_examples": code_free["examples"],
    }


def train(arm: str, out_dir: Path, steps: int) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, math_sha = math_task.load_manifest()
    _, code_sha = code_task.load_manifest()
    if math_task.file_sha(PARENT) != EXPECTED_PARENT:
        raise RuntimeError("C27 selected math checkpoint changed")
    device = math_task.choose_device()
    model, config, original_parent = math_task.start_model(device)
    saved = torch.load(PARENT, map_location="cpu", weights_only=True)
    if saved["task_manifest_sha256"] != math_sha or saved["start_model_identity"] != original_parent:
        raise RuntimeError("C27 selected math checkpoint identity mismatch")
    model.load_state_dict(saved["model"])
    anchor = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    importance = None
    if arm == "importance":
        if math_task.file_sha(IMPORTANCE_PATH) != EXPECTED_IMPORTANCE:
            raise RuntimeError("Stored math importance changed")
        table = torch.load(IMPORTANCE_PATH, map_location="cpu", weights_only=True)
        if table["math_checkpoint_sha256"] != EXPECTED_PARENT or table["math_manifest_sha256"] != math_sha:
            raise RuntimeError("Stored math importance source mismatch")
        importance = {name: value.to(device) for name, value in table["importance"].items()}
        if set(importance) != set(anchor):
            raise RuntimeError("Importance parameter names changed")
    tokenizer = Tokenizer.from_file(str(math_task.TOKENIZER_FILE))
    math_train = math_task.load_split("train")
    code_train = code_task.load_split("train")
    math_val = math_task.load_split("validation")
    code_val = code_task.load_split("validation")
    math_rng = np.random.default_rng(SEED + 1)
    code_rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    n_parameters = sum(p.numel() for p in model.parameters())
    history = [{"step": 0, "validation": validate(model, tokenizer, device, math_val, code_val)}]
    seen = {"code_response_targets": 0, "math_replay_response_targets": 0}
    for step in range(1, steps + 1):
        selected_code = code_rng.integers(0, len(code_train), size=16)
        code_batch = [code_train[int(index)] for index in selected_code]
        model.train()
        # Same code-side random stream at every step in all arms.
        torch.manual_seed(SEED + step)
        code_loss = mean_nll(model, code_batch, code_task.batch, tokenizer, device)
        seen["code_response_targets"] += sum(len(tokenizer.encode(row["answer"], add_special_tokens=False).ids) + 1
                                             for row in code_batch)
        objective = code_loss
        if arm == "replay":
            selected_math = math_rng.integers(0, len(math_train), size=16)
            math_batch = [math_train[int(index)] for index in selected_math]
            math_loss = mean_nll(model, math_batch, math_task.batch, tokenizer, device)
            objective = 0.5 * code_loss + 0.5 * math_loss
            seen["math_replay_response_targets"] += 4 * len(math_batch)
        elif arm == "importance":
            weighted_distance = sum(
                (importance[name] * (parameter - anchor[name]).square()).sum()
                for name, parameter in model.named_parameters()
            ) / n_parameters
            objective = code_loss + 0.5 * LAMBDA * weighted_distance
        if not torch.isfinite(objective):
            raise RuntimeError(f"Nonfinite objective in {arm} at step {step}")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == steps:
            current = validate(model, tokenizer, device, math_val, code_val)
            history.append({"step": step, "validation": current, "last_code_train_loss": float(code_loss.detach()),
                            "last_total_train_objective": float(objective.detach())})
            math_task.write_json(out_dir / f"progress_step_{step:04d}.json",
                                 {"arm": arm, "steps_completed": step, "history": history})
            print(arm, step, "math", current["old_math_validation_exact_and_eos"],
                  "code", current["new_code_validation_symbolically_equivalent"], flush=True)
    final_checkpoint = out_dir / "final.pt"
    torch.save({
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "math_parent_sha256": EXPECTED_PARENT, "importance_sha256": EXPECTED_IMPORTANCE if importance else None,
        "math_manifest_sha256": math_sha, "code_manifest_sha256": code_sha,
        "arm": arm, "steps": steps, "config": config.__dict__,
    }, final_checkpoint)
    report = {
        "scope": "Sequential B after previously learned A on same C19-C25-C27 4.93M weights; old C27 specialist tests already opened, selection/report here use validation",
        "arm": arm, "math_parent_sha256": EXPECTED_PARENT,
        "importance_sha256": EXPECTED_IMPORTANCE if importance else None,
        "math_manifest_sha256": math_sha, "code_manifest_sha256": code_sha,
        "training": {"steps": steps, "code_batch_each_step": 16, "math_replay_batch_each_step": 16 if arm == "replay" else 0,
                     "learning_rate": LR, "seed": SEED,
                     "importance_penalty_lambda": LAMBDA if arm == "importance" else None,
                     "code_sampling_identical_between_arms": True,
                     "weight_replay_means": [0.5, 0.5] if arm == "replay" else None},
        "history": history, "targets_seen": seen,
        "fixed_terminal_step": steps,
        "checkpoint": str(final_checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": math_task.file_sha(final_checkpoint),
        "device": str(device),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_test_status": "C27 math/code test rows previously opened; no fresh blind-test claim",
    }
    math_task.write_json(out_dir / "train.json", report)
    final = history[-1]["validation"]
    print(json.dumps({
        "arm": arm, "step": steps,
        "math_validation_exact": final["old_math_validation_exact_and_eos"],
        "code_validation_symbolic": final["new_code_validation_symbolically_equivalent"],
        "math_validation_nll": final["old_math_validation_nll"]["nll"],
        "code_validation_nll": final["new_code_validation_nll"]["nll"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("plain", "replay", "importance"))
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    train(args.arm, ROOT / args.out_dir, args.steps)
