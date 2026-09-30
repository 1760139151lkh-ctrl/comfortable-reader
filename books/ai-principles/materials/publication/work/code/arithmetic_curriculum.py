"""Response-only arithmetic SFT on the book's own C19-to-C25 causal LM.

All 8192 output tokens remain available at generation. Results are exact only
when the free output has the requested three digits and EOS. The official
GSM8K test file is not read by this program.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from evaluate_dolly_sft import load_run, selected_model
from train_dolly_sft import (
    ROOT, TOKENIZER_FILE, choose_device, file_sha, load_base,
    load_sft_manifest,
)


DATA = ROOT / "work/data/c27_digit_addition"
START_RUN = ROOT / "work/runs/c25_full_mix_trial"
BOS, EOS, PAD = 1, 2, 0
SEED = 20260924


def write_json(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preserve existing result: {path}")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prompt(tens: int, units: int, addend: int, *, paraphrase: bool = False,
           hundreds: int = 0) -> str:
    if paraphrase:
        instruction = (
            f"Start with digits {hundreds}, {tens}, {units} in hundreds, tens, units order. "
            f"Increase the number by {addend}. Answer with three spaced digits."
        )
    else:
        instruction = (
            f"A number has hundreds digit {hundreds}, tens digit {tens}, "
            f"and units digit {units}. Add {addend}. "
            "Write the result as three digits separated by spaces."
        )
    return "Instruction:\n" + instruction + "\nResponse:\n"


def answer(hundreds: int, tens: int, units: int, addend: int) -> str:
    value = 100 * hundreds + 10 * tens + units + addend
    if value >= 1000:
        raise ValueError("planned answer exceeds three digits")
    return " ".join(f"{value:03d}")


def prepare() -> None:
    if DATA.exists() and any(DATA.iterdir()):
        raise FileExistsError(f"preserve existing data: {DATA}")
    DATA.mkdir(parents=True)
    strata: dict[str, list[tuple[int, int, int]]] = {}
    for tens in range(10):
        for units in range(10):
            for addend in range(10):
                carry = int(units + addend >= 10)
                hundred_carry = int(tens == 9 and carry)
                key = f"units_carry={carry};hundred_carry={hundred_carry}"
                strata.setdefault(key, []).append((tens, units, addend))
    groups: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for key, triples in sorted(strata.items()):
        triples.sort(key=lambda row: hashlib.sha256(
            f"C27:{row[0]}:{row[1]}:{row[2]}".encode("ascii")
        ).digest())
        total = len(triples)
        train_cut = round(total * 0.70)
        val_cut = round(total * 0.85)
        for index, (tens, units, addend) in enumerate(triples):
            split = "train" if index < train_cut else "validation" if index < val_cut else "test"
            groups[split].append({
                "tens": tens, "units": units, "addend": addend,
                "prompt": prompt(tens, units, addend),
                "answer": answer(0, tens, units, addend),
                "stratum": key,
            })
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    lengths = []
    for split, records in groups.items():
        records.sort(key=lambda row: (row["tens"], row["units"], row["addend"]))
        for row in records:
            x = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            y = tokenizer.encode(row["answer"], add_special_tokens=False).ids
            lengths.append(1 + len(x) + len(y))
            if len(y) != 3 or 1 + len(x) + len(y) > 128:
                raise RuntimeError("answer encoding or context changed")
        write_json(DATA / f"{split}.json", records)
    manifest = {
        "task": "given separated hundreds/tens/units digits and a one-digit addend, emit three spaced decimal digits",
        "source": "author-defined deterministic arithmetic; no external model or user labels",
        "old_model": "C19 random WikiText pretraining, C25 full-mix Dolly SFT checkpoint",
        "tokenizer_sha256": file_sha(TOKENIZER_FILE),
        "context_limit": 128,
        "prompt_answer_sequence": "BOS + prompt + three response BPE tokens + EOS; only response/EOS targets scored",
        "split_rule": "SHA ordering within units-carry and hundred-carry strata, 70/15/15 percent",
        "counts": {key: len(records) for key, records in groups.items()},
        "prompt_plus_three_response_length_min_max": [min(lengths), max(lengths)],
        "splits": {key: file_sha(DATA / f"{key}.json") for key in groups},
        "test_policy": "train and selection never read test.json; evaluate opens once after chosen checkpoint",
    }
    write_json(DATA / "manifest.json", manifest)
    print("prepared", manifest["counts"], "length", manifest["prompt_plus_three_response_length_min_max"])


def load_manifest() -> tuple[dict, str]:
    path = DATA / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if file_sha(TOKENIZER_FILE) != manifest["tokenizer_sha256"]:
        raise RuntimeError("tokenizer changed")
    for split, expected in manifest["splits"].items():
        if file_sha(DATA / f"{split}.json") != expected:
            raise RuntimeError(f"{split} changed")
    return manifest, file_sha(path)


def load_split(split: str) -> list[dict]:
    return json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))


def start_model(device: torch.device):
    _, sft_sha = load_sft_manifest()
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


def batch(records: list[dict], tokenizer: Tokenizer, device: torch.device):
    streams = []
    prompt_sizes = []
    for record in records:
        p = tokenizer.encode(record["prompt"], add_special_tokens=False).ids
        a = tokenizer.encode(record["answer"], add_special_tokens=False).ids
        streams.append([BOS] + p + a + [EOS])
        prompt_sizes.append(len(p))
    width = max(len(stream) - 1 for stream in streams)
    ids = torch.full((len(records), width), PAD, dtype=torch.long, device=device)
    targets = torch.full((len(records), width), -100, dtype=torch.long, device=device)
    real = torch.zeros((len(records), width), dtype=torch.bool, device=device)
    for index, stream in enumerate(streams):
        size = len(stream) - 1
        ids[index, :size] = torch.tensor(stream[:-1], device=device)
        targets[index, :size] = torch.tensor(stream[1:], device=device)
        targets[index, :prompt_sizes[index]] = -100
        real[index, :size] = True
    if int((targets != -100).sum()) != 4 * len(records):
        raise RuntimeError("expected three response tokens and EOS for every row")
    return ids, targets, real


def prompt_batch(records: list[dict], tokenizer: Tokenizer, device: torch.device):
    rows = [[BOS] + tokenizer.encode(record["prompt"], add_special_tokens=False).ids
            for record in records]
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), PAD, dtype=torch.long, device=device)
    real = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    last = []
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, device=device)
        real[index, :len(row)] = True
        last.append(len(row) - 1)
    return ids, real, torch.tensor(last, dtype=torch.long, device=device)


def autocast_for(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def teacher_nll(model, records: list[dict], tokenizer: Tokenizer, device: torch.device) -> dict:
    model.eval()
    total = 0.0
    count = 0
    for offset in range(0, len(records), 64):
        part = records[offset:offset + 64]
        ids, targets, real = batch(part, tokenizer, device)
        with autocast_for(device):
            logits = model(ids, real)
        total += float(F.cross_entropy(
            logits.float().transpose(1, 2), targets,
            ignore_index=-100, reduction="sum",
        ))
        count += int((targets != -100).sum())
    return {"nll": total / count, "targets": count}


@torch.no_grad()
def free_generate(model, records: list[dict], tokenizer: Tokenizer,
                  device: torch.device, max_new_tokens: int = 8) -> dict:
    model.eval()
    exact = 0
    syntactic_three = 0
    numerical = 0
    eos_count = 0
    examples = []
    pattern = re.compile(r"^[0-9] [0-9] [0-9]$")
    for offset in range(0, len(records), 32):
        part = records[offset:offset + 32]
        ids, real, last = prompt_batch(part, tokenizer, device)
        generated: list[list[int]] = [[] for _ in part]
        finished = torch.zeros(len(part), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            with autocast_for(device):
                logits = model(ids, real)[torch.arange(len(part), device=device), last]
            chosen = logits.float().argmax(dim=-1)
            active = ~finished
            for index in range(len(part)):
                if bool(active[index]):
                    generated[index].append(int(chosen[index]))
            finished = finished | (chosen == EOS)
            if bool(finished.all()):
                break
            extension = torch.where(finished, PAD, chosen)
            ids = torch.cat((ids, extension[:, None]), dim=1)
            real = torch.cat((real, (~finished)[:, None]), dim=1)
            last = last + (~finished).long()
        for record, tokens in zip(part, generated):
            has_eos = EOS in tokens
            eos_count += int(has_eos)
            clean = tokens[:tokens.index(EOS)] if has_eos else tokens
            output = tokenizer.decode(clean).strip()
            syntax = bool(pattern.fullmatch(output))
            syntactic_three += int(syntax)
            numerical += int(syntax and int(output.replace(" ", "")) ==
                             int(record["answer"].replace(" ", "")))
            exact += int(has_eos and output == record["answer"])
            if len(examples) < 5:
                examples.append({
                    "input_digits": f"0{record['tens']}{record['units']}+{record['addend']}",
                    "expected": record["answer"], "generated": output,
                    "eos": has_eos, "token_ids": tokens,
                })
    return {
        "records": len(records), "exact_answer_and_eos": exact,
        "syntactically_three_spaced_digits": syntactic_three,
        "numerically_correct_if_syntax_valid": numerical,
        "eos_within_limit": eos_count, "max_new_tokens": max_new_tokens,
        "examples": examples,
    }


def probe(out: Path) -> None:
    manifest, manifest_sha = load_manifest()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    device = choose_device()
    model, cfg, identity = start_model(device)
    train = load_split("train")
    validation = load_split("validation")
    first = train[0]
    ids, targets, real = batch([first], tokenizer, device)
    result = {
        "scope": "C25 full-mix start before arithmetic SFT; test unopened",
        "task_manifest_sha256": manifest_sha, "model_identity": identity,
        "first_train_example": {
            "input_digits": f"0{first['tens']}{first['units']}+{first['addend']}",
            "prompt": first["prompt"], "answer": first["answer"],
            "real_input_positions": int(real.sum()),
            "scored_targets": int((targets != -100).sum()),
            "first_scored_input_index": int(torch.where(targets[0] != -100)[0][0]),
            "first_target_id": int(targets[0, torch.where(targets[0] != -100)[0][0]]),
        },
        "baseline_validation_nll": teacher_nll(model, validation, tokenizer, device),
        "baseline_validation_free": free_generate(model, validation, tokenizer, device),
        "context_limit": cfg.context, "data_counts": manifest["counts"],
    }
    write_json(out, result)
    print("probe NLL", result["baseline_validation_nll"]["nll"],
          "free exact", result["baseline_validation_free"]["exact_answer_and_eos"])


def train(out_dir: Path, steps: int, learning_rate: float) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha = load_manifest()
    training = load_split("train")
    validation = load_split("validation")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    device = choose_device()
    model, cfg, identity = start_model(device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    base_nll = teacher_nll(model, validation, tokenizer, device)
    base_free = free_generate(model, validation, tokenizer, device)
    best_key = (base_free["exact_answer_and_eos"], -base_nll["nll"])
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"step": 0, "validation_nll": base_nll,
                "validation_free": base_free}]
    first_update = None
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(training), size=32)
        part = [training[int(index)] for index in indices]
        ids, targets, real = batch(part, tokenizer, device)
        model.train()
        with autocast_for(device):
            logits = model(ids, real)
        total = F.cross_entropy(
            logits.float().transpose(1, 2), targets,
            ignore_index=-100, reduction="sum",
        )
        scored = int((targets != -100).sum())
        loss = total / scored
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        before = model.output.weight.detach().clone() if step == 1 else None
        optimizer.step()
        if step == 1:
            first_update = {
                "first_batch_records": len(part),
                "first_batch_scored_targets": scored,
                "first_batch_nll": float(loss.detach()),
                "gradient_norm_before_clip": grad_norm,
                "output_weight_change_norm": float(torch.linalg.vector_norm(
                    model.output.weight.detach() - before
                )),
            }
        if step % 100 == 0:
            current_nll = teacher_nll(model, validation, tokenizer, device)
            current_free = free_generate(model, validation, tokenizer, device)
            history.append({"step": step, "validation_nll": current_nll,
                            "validation_free": current_free})
            key = (current_free["exact_answer_and_eos"], -current_nll["nll"])
            if key > best_key:
                best_key = key
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print("step", step, "val_nll", round(current_nll["nll"], 4),
                  "free_exact", current_free["exact_answer_and_eos"],
                  "format", current_free["syntactically_three_spaced_digits"],
                  flush=True)
    model.load_state_dict(best_state)
    checkpoint = {
        "model": {name: value.detach().cpu().clone()
                  for name, value in model.state_dict().items()},
        "selected_step": best_step, "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity, "config": cfg.__dict__,
    }
    best_path = out_dir / "best.pt"
    torch.save(checkpoint, best_path)
    report = {
        "scope": "C25 full-mix model response-only SFT on author-generated 3-digit formatted addition; full-vocabulary free greedy output",
        "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity,
        "device": str(device), "torch_version": torch.__version__,
        "training": {
            "steps": steps, "batch_size": 32,
            "learning_rate": learning_rate, "betas": [0.9, 0.95],
            "weight_decay": 0.01, "gradient_clip_norm": 1.0,
            "selection": "highest validation free exact+EOS, tie by lower teacher NLL, checked each 100 steps",
        },
        "first_update": first_update, "history": history,
        "selected_step": best_step,
        "selected_validation_nll": teacher_nll(model, validation, tokenizer, device),
        "selected_validation_free": free_generate(model, validation, tokenizer, device),
        "checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": file_sha(best_path),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "test_opened": False,
    }
    write_json(out_dir / "train.json", report)
    print("selected", best_step, best_key)


def evaluate(run_dir: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"test already opened: {out}")
    manifest, manifest_sha = load_manifest()
    report = json.loads((run_dir / "train.json").read_text(encoding="utf-8"))
    checkpoint_path = ROOT / report["checkpoint"]
    if (report["task_manifest_sha256"] != manifest_sha
            or report["test_opened"] is not False
            or file_sha(checkpoint_path) != report["checkpoint_sha256"]):
        raise RuntimeError("training result identity changed")
    device = choose_device()
    baseline, cfg, identity = start_model(device)
    if identity != report["start_model_identity"]:
        raise RuntimeError("starting C25 checkpoint changed")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (state["task_manifest_sha256"] != manifest_sha
            or state["start_model_identity"] != identity
            or state["selected_step"] != report["selected_step"]):
        raise RuntimeError("selected checkpoint changed")
    trained = copy.deepcopy(baseline)
    trained.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    # No test records were read by probe or train.
    test = load_split("test")
    paraphrase = [{
        **row, "prompt": prompt(row["tens"], row["units"], row["addend"],
                               paraphrase=True)
    } for row in test]
    longer = [{
        "tens": t, "units": u, "addend": (7 * t + 3 * u) % 10,
        "prompt": prompt(t, u, (7 * t + 3 * u) % 10, hundreds=1),
        "answer": answer(1, t, u, (7 * t + 3 * u) % 10),
    } for t in range(10) for u in range(10)]
    results = {}
    for name, model in (("start_c25_full_mix", baseline), ("selected_math_sft", trained)):
        results[name] = {
            "test_same_template_nll": teacher_nll(model, test, tokenizer, device),
            "test_same_template_free": free_generate(model, test, tokenizer, device),
            "same_numbers_paraphrase_free": free_generate(model, paraphrase, tokenizer, device),
            "one_more_hundred_ood_free": free_generate(model, longer, tokenizer, device),
        }
    output = {
        "scope": "test opened once after validation-selected checkpoint; free full-vocab greedy with EOS requirement; wording and hundreds-digit changes predeclared",
        "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity,
        "selected_checkpoint_sha256": report["checkpoint_sha256"],
        "selected_step": report["selected_step"],
        "results": results,
    }
    write_json(out, output)
    print("test free exact", results["start_c25_full_mix"]["test_same_template_free"]["exact_answer_and_eos"],
          "to", results["selected_math_sft"]["test_same_template_free"]["exact_answer_and_eos"],
          "of", len(test))


def main() -> None:
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "probe", "train", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=Path("work/data/c27_digit_addition"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("work/runs/c27_digit_addition_trial"))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    args = parser.parse_args()
    DATA = ROOT / args.data_dir
    if args.mode == "prepare":
        prepare()
    elif args.mode == "probe":
        probe(ROOT / args.out if args.out else
              ROOT / "work/results/c27_digit_initial_probe.json")
    elif args.mode == "train":
        train(ROOT / args.run_dir, args.steps, args.learning_rate)
    else:
        evaluate(ROOT / args.run_dir,
                 ROOT / args.out if args.out else
                 ROOT / "work/results/c27_digit_first_test.json")


if __name__ == "__main__":
    main()
