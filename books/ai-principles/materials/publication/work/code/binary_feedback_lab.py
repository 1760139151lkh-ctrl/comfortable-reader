"""A small, honest result-feedback loop on the book's C19-to-C25 model.

The host restricts each response to the single BPE token A or B. This is a
different policy from unrestricted language generation, and the reports retain
that distinction. The test split is opened only by the evaluate command.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from evaluate_dolly_sft import load_run, selected_model
from train_dolly_sft import (
    ROOT, TOKENIZER_FILE, choose_device, file_sha, load_base,
    load_sft_manifest,
)


DATA = ROOT / "work/data/c26_binary_digits"
START_RUN = ROOT / "work/runs/c25_full_mix_trial"
BOS = 1
PAD = 0
SEED = 20260924
ACTION_TEXT = ("A", "B")
TRAIN_STEPS = 200
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
REFERENCE_KL_WEIGHT = 0.005
VALIDATE_EVERY = 20


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preserve existing result: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prompt(tens: int, units: int, paraphrase: bool = False,
           separate_digits: bool = False) -> str:
    if separate_digits and paraphrase:
        instruction = (
            f"The tens digit is {tens}; the units digit is {units}. "
            "Write A when the whole number is even, or B otherwise."
        )
    elif separate_digits:
        instruction = (
            f"Tens digit: {tens}. Units digit: {units}. "
            "Is the number even? Reply A for even and B for odd."
        )
    elif paraphrase:
        instruction = (
            f"Check {tens}{units}: is its final digit even? "
            "Write A if it is, otherwise B."
        )
    else:
        instruction = (
            f"For the two-digit number {tens}{units}, is its last digit even? "
            "Reply A for even and B for odd."
        )
    return "Instruction:\n" + instruction + "\nResponse:\n"


def correct_action(item: dict) -> int:
    return 0 if item["units"] % 2 == 0 else 1


def prepare(separate_digits: bool = False) -> None:
    if DATA.exists() and any(DATA.iterdir()):
        raise FileExistsError(f"preserve existing task split: {DATA}")
    DATA.mkdir(parents=True)
    splits = {"train": [], "validation": [], "test": []}
    for units in range(10):
        # Seven, one, and two different two-digit strings per units digit.
        ordered_tens = sorted(
            range(10),
            key=lambda tens: hashlib.sha256(
                f"C26:{tens}:{units}".encode("ascii")
            ).digest(),
        )
        for rank, tens in enumerate(ordered_tens):
            split = "train" if rank < 7 else "validation" if rank == 7 else "test"
            splits[split].append({
                "tens": tens,
                "units": units,
                "prompt": prompt(tens, units, separate_digits=separate_digits),
            })
    for split in splits:
        splits[split].sort(key=lambda item: (item["tens"], item["units"]))
        write_json(DATA / f"{split}.json", splits[split])
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    action_ids = [tokenizer.encode(word, add_special_tokens=False).ids for word in ACTION_TEXT]
    if any(len(ids) != 1 for ids in action_ids):
        raise RuntimeError("A and B are no longer one BPE token each")
    lengths = [len(tokenizer.encode(item["prompt"], add_special_tokens=False).ids) + 1
               for group in splits.values() for item in group]
    if max(lengths) > 128:
        raise RuntimeError("task prompt exceeds C19 context")
    manifest = {
        "purpose": "two-choice result-feedback mechanics, not open-ended math or code",
        "prompt_rule": "A if units digit even, B if odd; verifier uses integer arithmetic",
        "prompt_style": "separated_digits" if separate_digits else "joined_number",
        "output_policy": "one sampled BPE token, restricted and renormalized to A/B",
        "base_model": "C19 random-pretrained then C25 full-mix SFT",
        "action_text": ACTION_TEXT,
        "action_token_ids": [ids[0] for ids in action_ids],
        "tokenizer_sha256": file_sha(TOKENIZER_FILE),
        "prompt_lengths_with_bos_min_max": [min(lengths), max(lengths)],
        "splits": {
            name: {"count": len(items), "sha256": file_sha(DATA / f"{name}.json")}
            for name, items in splits.items()
        },
        "selection": "training samples/validation expected reward only; test and paraphrase after checkpoint fixed",
    }
    write_json(DATA / "manifest.json", manifest)
    print("prepared", {k: len(v) for k, v in splits.items()}, "action ids", manifest["action_token_ids"])


def load_manifest() -> tuple[dict, str]:
    path = DATA / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if file_sha(TOKENIZER_FILE) != manifest["tokenizer_sha256"]:
        raise RuntimeError("tokenizer changed")
    for split, entry in manifest["splits"].items():
        if file_sha(DATA / f"{split}.json") != entry["sha256"]:
            raise RuntimeError(f"{split} split changed")
    return manifest, file_sha(path)


def load_split(split: str) -> list[dict]:
    return json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))


def start_model(device: torch.device):
    sft_manifest, sft_sha = load_sft_manifest()
    _ = sft_manifest
    _, _, base = load_base()
    report, checkpoint = load_run(
        START_RUN, "full-mix", sft_sha, base["c19_checkpoint_sha256"]
    )
    model, cfg = selected_model("full-mix", checkpoint, device)
    return model, cfg, {
        "c19_checkpoint_sha256": base["c19_checkpoint_sha256"],
        "c25_sft_manifest_sha256": sft_sha,
        "c25_full_mix_checkpoint_sha256": report["checkpoint_sha256"],
    }


def encode_batch(items: list[dict], tokenizer: Tokenizer, device: torch.device):
    rows = [[BOS] + tokenizer.encode(item["prompt"], add_special_tokens=False).ids
            for item in items]
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), PAD, dtype=torch.long, device=device)
    real = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    last = []
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        real[i, :len(row)] = True
        last.append(len(row) - 1)
    return ids, real, torch.tensor(last, dtype=torch.long, device=device)


def first_action_logits(model, items: list[dict], tokenizer: Tokenizer,
                        action_ids: list[int], device: torch.device):
    ids, real, last = encode_batch(items, tokenizer, device)
    all_logits = model(ids, real)[torch.arange(len(items), device=device), last]
    return all_logits[:, action_ids], all_logits


@torch.no_grad()
def measure(model, items: list[dict], tokenizer: Tokenizer,
            action_ids: list[int], device: torch.device) -> dict:
    model.eval()
    probability_correct = []
    greedy_correct = 0
    raw_mass = []
    full_greedy_action = 0
    examples = []
    for offset in range(0, len(items), 32):
        part = items[offset:offset + 32]
        two_logits, all_logits = first_action_logits(model, part, tokenizer, action_ids, device)
        two_prob = torch.softmax(two_logits.float(), dim=-1)
        raw_prob = torch.softmax(all_logits.float(), dim=-1)
        for i, item in enumerate(part):
            target = correct_action(item)
            probability_correct.append(float(two_prob[i, target]))
            greedy_correct += int(int(two_prob[i].argmax()) == target)
            raw_mass.append(float(raw_prob[i, action_ids].sum()))
            top_id = int(raw_prob[i].argmax())
            full_greedy_action += int(top_id in action_ids)
            if len(examples) < 4:
                examples.append({
                    "number": f"{item['tens']}{item['units']}",
                    "expected": ACTION_TEXT[target],
                    "restricted_p_A": float(two_prob[i, 0]),
                    "raw_full_vocab_p_A_or_B": float(raw_prob[i, action_ids].sum()),
                    "unrestricted_top_token": tokenizer.decode([top_id]),
                })
    return {
        "records": len(items),
        "expected_reward_restricted": float(np.mean(probability_correct)),
        "greedy_correct_restricted": greedy_correct,
        "raw_full_vocab_action_mass_mean": float(np.mean(raw_mass)),
        "full_vocab_greedy_first_token_A_or_B": full_greedy_action,
        "examples": examples,
    }


def probe(out: Path) -> None:
    manifest, manifest_sha = load_manifest()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    device = choose_device()
    model, cfg, identity = start_model(device)
    train = load_split("train")
    validation = load_split("validation")
    report = {
        "scope": "initial C25 model on two-choice prompts; no test split, no update",
        "task_manifest_sha256": manifest_sha,
        "model_identity": identity,
        "context": cfg.context,
        "train": measure(model, train, tokenizer, manifest["action_token_ids"], device),
        "validation": measure(model, validation, tokenizer, manifest["action_token_ids"], device),
        "zero_reward_gradient": (
            "If every permitted action has reward zero, the exact expected reward is constant. "
            "With baseline zero, reward times log-probability gives a zero policy gradient; "
            "reference KL can still pull toward its reference, but supplies no task success information."
        ),
    }
    write_json(out, report)
    print("probe validation", report["validation"]["expected_reward_restricted"],
          "raw mass", report["validation"]["raw_full_vocab_action_mass_mean"])


def train(out_dir: Path, *, kl_weight: float = REFERENCE_KL_WEIGHT,
          balanced: bool = False, steps: int = TRAIN_STEPS,
          learning_rate: float = LEARNING_RATE) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha = load_manifest()
    train_items = load_split("train")
    validation = load_split("validation")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    action_ids = manifest["action_token_ids"]
    device = choose_device()
    model, cfg, identity = start_model(device)
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    baseline_measure = measure(model, validation, tokenizer, action_ids, device)
    best_value = baseline_measure["expected_reward_restricted"]
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"step": 0, "validation": baseline_measure}]
    first_update = None
    even_indices = np.array([i for i, item in enumerate(train_items)
                             if correct_action(item) == 0])
    odd_indices = np.array([i for i, item in enumerate(train_items)
                            if correct_action(item) == 1])
    for step in range(1, steps + 1):
        if balanced:
            indices = np.concatenate((
                rng.choice(even_indices, size=BATCH_SIZE // 2, replace=True),
                rng.choice(odd_indices, size=BATCH_SIZE // 2, replace=True),
            ))
            rng.shuffle(indices)
        else:
            indices = rng.integers(0, len(train_items), size=BATCH_SIZE)
        part = [train_items[int(index)] for index in indices]
        model.eval()  # no dropout change between sampled policy and recomputation
        with torch.no_grad():
            old_logits, _ = first_action_logits(model, part, tokenizer, action_ids, device)
            old_log_probs = torch.log_softmax(old_logits.float(), dim=-1)
            sampled = torch.distributions.Categorical(logits=old_logits.float()).sample()
            ref_logits, _ = first_action_logits(reference, part, tokenizer, action_ids, device)
            ref_log_probs = torch.log_softmax(ref_logits.float(), dim=-1)
        target = torch.tensor([correct_action(item) for item in part], device=device)
        reward = (sampled == target).float()
        # Every other action is independent of this sample's action at fixed theta.
        other_mean = (reward.sum() - reward) / (BATCH_SIZE - 1)
        advantage = (reward - other_mean).detach()
        current_logits, _ = first_action_logits(model, part, tokenizer, action_ids, device)
        current_log_probs = torch.log_softmax(current_logits.float(), dim=-1)
        sampled_logp = current_log_probs.gather(1, sampled[:, None]).squeeze(1)
        current_p = current_log_probs.exp()
        exact_two_action_kl = (current_p * (current_log_probs - ref_log_probs)).sum(-1).mean()
        reward_loss = -(advantage * sampled_logp).mean()
        loss = reward_loss + kl_weight * exact_two_action_kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        before = model.output.weight.detach()[action_ids].clone() if step == 1 else None
        optimizer.step()
        if step == 1:
            after = model.output.weight.detach()[action_ids]
            first_update = {
                "numbers": [f"{x['tens']}{x['units']}" for x in part],
                "sampled_action": [ACTION_TEXT[int(x)] for x in sampled],
                "reward": reward.tolist(),
                "leave_one_out_baseline": other_mean.tolist(),
                "advantage": advantage.tolist(),
                "sampled_action_old_log_probability": old_log_probs.gather(
                    1, sampled[:, None]
                ).squeeze(1).tolist(),
                "policy_gradient_loss": float(reward_loss.detach()),
                "reference_kl_at_old_weights": float(exact_two_action_kl.detach()),
                "gradient_norm_before_clip": grad_norm,
                "output_A_B_row_change_norm": float(torch.linalg.vector_norm(after - before)),
            }
        if step % VALIDATE_EVERY == 0:
            validation_measure = measure(model, validation, tokenizer, action_ids, device)
            history.append({
                "step": step,
                "sampled_batch_reward": float(reward.mean()),
                "reference_kl_in_batch_before_step": float(exact_two_action_kl.detach()),
                "validation": validation_measure,
            })
            value = validation_measure["expected_reward_restricted"]
            if value > best_value:
                best_value = value
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print("step", step, "validation_expected_reward", round(value, 4),
                  "greedy", validation_measure["greedy_correct_restricted"], flush=True)
    model.load_state_dict(best_state)
    checkpoint = {
        "model": {key: tensor.detach().cpu().clone()
                  for key, tensor in model.state_dict().items()},
        "selected_step": best_step,
        "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity,
        "config": cfg.__dict__,
        "action_token_ids": action_ids,
    }
    checkpoint_path = out_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "scope": "on-policy one-token restricted A/B REINFORCE, exact two-action reference KL; not PPO/GRPO or free generation",
        "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity,
        "device": str(device),
        "torch_version": torch.__version__,
        "action_token_ids": action_ids,
        "training": {
            "steps": steps, "batch_size": BATCH_SIZE,
            "learning_rate": learning_rate,
            "balanced_prompt_batch": balanced,
            "reference_kl_weight": kl_weight,
            "baseline": "mean reward of other 15 independently sampled actions",
            "selection": "max validation mean exact P(correct) every 20 steps",
        },
        "first_update": first_update,
        "history": history,
        "selected_step": best_step,
        "selected_validation": measure(model, validation, tokenizer, action_ids, device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": file_sha(checkpoint_path),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "test_opened": False,
    }
    write_json(out_dir / "train.json", report)
    print("selected step", best_step, "val", best_value)


def evaluate(run_dir: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"test already opened: {out}")
    manifest, manifest_sha = load_manifest()
    report = json.loads((run_dir / "train.json").read_text(encoding="utf-8"))
    checkpoint_path = ROOT / report["checkpoint"]
    if (
        report["task_manifest_sha256"] != manifest_sha
        or report["test_opened"] is not False
        or file_sha(checkpoint_path) != report["checkpoint_sha256"]
    ):
        raise RuntimeError("training identity or checkpoint changed")
    device = choose_device()
    base, cfg, identity = start_model(device)
    if report["start_model_identity"] != identity:
        raise RuntimeError("C25 starting model changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["task_manifest_sha256"] != manifest_sha
        or checkpoint["start_model_identity"] != identity
        or checkpoint["selected_step"] != report["selected_step"]
        or checkpoint["action_token_ids"] != manifest["action_token_ids"]
    ):
        raise RuntimeError("selected model identity mismatch")
    trained = copy.deepcopy(base)
    trained.load_state_dict(checkpoint["model"])
    trained.eval()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    # The only command that opens these source numbers is evaluate.
    test = load_split("test")
    changed_words = [
        {**item, "prompt": prompt(
            item["tens"], item["units"], paraphrase=True,
            separate_digits=manifest["prompt_style"] == "separated_digits",
        )}
        for item in test
    ]
    result = {
        "scope": "test opened once after validation-selected checkpoint; same-template held-out numbers and predeclared paraphrase",
        "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity,
        "selected_checkpoint_sha256": report["checkpoint_sha256"],
        "selected_step": report["selected_step"],
        "same_template": {
            "start": measure(base, test, tokenizer, manifest["action_token_ids"], device),
            "trained": measure(trained, test, tokenizer, manifest["action_token_ids"], device),
        },
        "different_wording_same_numbers": {
            "start": measure(base, changed_words, tokenizer, manifest["action_token_ids"], device),
            "trained": measure(trained, changed_words, tokenizer, manifest["action_token_ids"], device),
        },
    }
    write_json(out, result)
    print("heldout expected reward", result["same_template"]["start"]["expected_reward_restricted"],
          "to", result["same_template"]["trained"]["expected_reward_restricted"])


def main() -> None:
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "probe", "train", "evaluate"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("work/runs/c26_binary_feedback_trial"))
    parser.add_argument("--kl-weight", type=float, default=REFERENCE_KL_WEIGHT)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--data-dir", type=Path, default=Path("work/data/c26_binary_digits"))
    parser.add_argument("--separate-digits", action="store_true")
    args = parser.parse_args()
    DATA = ROOT / args.data_dir
    if args.mode == "prepare":
        prepare(separate_digits=args.separate_digits)
    elif args.mode == "probe":
        probe((ROOT / args.out) if args.out else ROOT / "work/results/c26_binary_initial_probe.json")
    elif args.mode == "train":
        train(ROOT / args.run_dir, kl_weight=args.kl_weight,
              balanced=args.balanced, steps=args.steps,
              learning_rate=args.learning_rate)
    else:
        evaluate(ROOT / args.run_dir,
                 ROOT / args.out if args.out else ROOT / "work/results/c26_binary_feedback_test.json")


if __name__ == "__main__":
    main()
