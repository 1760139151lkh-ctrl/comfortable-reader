"""Short Python-expression generation with syntax, tests, and exact polynomial checks.

Generated text is never exec'd or eval'd. An explicit AST whitelist interprets
addition and multiplication of one variable and integers. This is a small
expression task, not HumanEval functions or repository repair.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from arithmetic_curriculum import (
    ROOT, TOKENIZER_FILE, BOS, EOS, PAD, autocast_for,
    choose_device, file_sha, prompt_batch, start_model, write_json,
)


DATA = ROOT / "work/data/c27_code_expression"
SEED = 20260924
VARIABLES = ("n", "x", "value")
FAMILIES = ("add", "multiply", "double_then_add")
VALIDATION_TEST_INPUTS = (-5, -2, 0, 1, 4, 9)
HELDOUT_TEST_INPUTS = (-11, -3, 2, 5, 13)


def prompt(family: str, variable: str, constant: int, *, paraphrase: bool = False) -> str:
    if family == "add":
        task = (f"Increase {variable} by {constant}."
                if paraphrase else f"Add {constant} to {variable}.")
    elif family == "multiply":
        task = (f"Scale {variable} by a factor of {constant}."
                if paraphrase else f"Multiply {variable} by {constant}.")
    elif family == "double_then_add":
        task = (f"Take twice {variable}, then increase it by {constant}."
                if paraphrase else f"Double {variable}, then add {constant}.")
    else:
        raise ValueError(family)
    return (
        "Instruction:\nWrite one Python arithmetic expression using variable "
        + variable + ". " + task + " Output only the expression.\nResponse:\n"
    )


def canonical(family: str, variable: str, constant: int) -> str:
    if family == "add":
        return f"{variable} + {constant}"
    if family == "multiply":
        return f"{variable} * {constant}"
    if family == "double_then_add":
        return f"2 * {variable} + {constant}"
    raise ValueError(family)


def target_polynomial(family: str, constant: int) -> dict[int, int]:
    if family == "add":
        return {0: constant, 1: 1}
    if family == "multiply":
        return {1: constant}
    if family == "double_then_add":
        return {0: constant, 1: 2}
    raise ValueError(family)


def prepare() -> None:
    if DATA.exists() and any(DATA.iterdir()):
        raise FileExistsError(f"preserve existing data: {DATA}")
    DATA.mkdir(parents=True)
    splits = {"train": [], "validation": [], "test": []}
    for family in FAMILIES:
        for variable in VARIABLES:
            constants = sorted(range(30), key=lambda value: hashlib.sha256(
                f"C27-code:{family}:{variable}:{value}".encode("utf-8")
            ).digest())
            for position, constant in enumerate(constants):
                split = "train" if position < 21 else "validation" if position < 25 else "test"
                splits[split].append({
                    "family": family, "variable": variable, "constant": constant,
                    "prompt": prompt(family, variable, constant),
                    "answer": canonical(family, variable, constant),
                })
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    longest = 0
    for split, records in splits.items():
        records.sort(key=lambda row: (
            row["family"], row["variable"], row["constant"]
        ))
        for row in records:
            size = 1 + len(tokenizer.encode(row["prompt"], add_special_tokens=False).ids)
            size += len(tokenizer.encode(row["answer"], add_special_tokens=False).ids)
            longest = max(longest, size)
            if size > 128:
                raise RuntimeError("C19 context overflow")
        write_json(DATA / f"{split}.json", records)
    manifest = {
        "task": "short Python arithmetic expression in one variable; add, multiply, double then add",
        "source": "author-generated exact specification, canonical demonstration expressions",
        "checkpoint_start": "C19 random WikiText pretraining then C25 Dolly full-mix SFT",
        "tokenizer_sha256": file_sha(TOKENIZER_FILE),
        "context_limit": 128, "max_sequence_length": longest,
        "splits": {name: {"records": len(records),
                          "sha256": file_sha(DATA / f"{name}.json")}
                   for name, records in splits.items()},
        "split_rule": "within each family and variable, SHA-order constants 21/4/5",
        "validation_inputs": VALIDATION_TEST_INPUTS,
        "heldout_test_inputs": HELDOUT_TEST_INPUTS,
        "evaluation": "free greedy code text; AST whitelist Add/Mult/Name/Int, finite tests, polynomial coefficients",
        "test_policy": "train/probe read only train and validation; evaluate opens test after validation-selected checkpoint",
    }
    write_json(DATA / "manifest.json", manifest)
    print("prepared", {name: len(records) for name, records in splits.items()},
          "longest", longest)


def load_manifest() -> tuple[dict, str]:
    path = DATA / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if file_sha(TOKENIZER_FILE) != data["tokenizer_sha256"]:
        raise RuntimeError("tokenizer changed")
    for split, entry in data["splits"].items():
        if file_sha(DATA / f"{split}.json") != entry["sha256"]:
            raise RuntimeError(f"{split} split changed")
    return data, file_sha(path)


def load_split(name: str) -> list[dict]:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def batch(records: list[dict], tokenizer: Tokenizer, device: torch.device):
    streams = []
    prompt_lengths = []
    for row in records:
        prefix = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
        response = tokenizer.encode(row["answer"], add_special_tokens=False).ids
        streams.append([BOS] + prefix + response + [EOS])
        prompt_lengths.append(len(prefix))
    width = max(len(x) - 1 for x in streams)
    ids = torch.full((len(records), width), PAD, dtype=torch.long, device=device)
    labels = torch.full((len(records), width), -100, dtype=torch.long, device=device)
    real = torch.zeros((len(records), width), dtype=torch.bool, device=device)
    for i, stream in enumerate(streams):
        size = len(stream) - 1
        ids[i, :size] = torch.tensor(stream[:-1], dtype=torch.long, device=device)
        labels[i, :size] = torch.tensor(stream[1:], dtype=torch.long, device=device)
        labels[i, :prompt_lengths[i]] = -100
        real[i, :size] = True
    return ids, labels, real


def polynomial_of_ast(node: ast.AST, variable: str) -> dict[int, int]:
    if isinstance(node, ast.Expression):
        return polynomial_of_ast(node.body, variable)
    if isinstance(node, ast.Name) and node.id == variable:
        return {1: 1}
    if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 99:
        return {0: node.value} if node.value else {}
    if isinstance(node, ast.BinOp):
        left = polynomial_of_ast(node.left, variable)
        right = polynomial_of_ast(node.right, variable)
        if isinstance(node.op, ast.Add):
            result = left.copy()
            for degree, value in right.items():
                result[degree] = result.get(degree, 0) + value
        elif isinstance(node.op, ast.Mult):
            result = {}
            for d1, v1 in left.items():
                for d2, v2 in right.items():
                    if d1 + d2 > 3:
                        raise ValueError("polynomial degree exceeds explicit checker")
                    result[d1 + d2] = result.get(d1 + d2, 0) + v1 * v2
        else:
            raise ValueError("only addition and multiplication allowed")
        return {degree: value for degree, value in result.items() if value}
    raise ValueError(f"AST node not allowed: {type(node).__name__}")


def polynomial_value(poly: dict[int, int], number: int) -> int:
    return sum(coefficient * number ** degree
               for degree, coefficient in poly.items())


def check_expression(text: str, row: dict, test_inputs: tuple[int, ...]) -> dict:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return {"parseable": False, "allowed_ast": False,
                "passes_finite_tests": False, "symbolically_equivalent": False}
    try:
        poly = polynomial_of_ast(tree, row["variable"])
    except ValueError:
        return {"parseable": True, "allowed_ast": False,
                "passes_finite_tests": False, "symbolically_equivalent": False}
    expected = target_polynomial(row["family"], row["constant"])
    expected = {degree: value for degree, value in expected.items() if value}
    finite = all(polynomial_value(poly, number) ==
                 polynomial_value(expected, number) for number in test_inputs)
    return {"parseable": True, "allowed_ast": True,
            "passes_finite_tests": finite,
            "symbolically_equivalent": poly == expected,
            "polynomial": poly}


@torch.no_grad()
def teacher_nll(model, records: list[dict], tokenizer: Tokenizer,
                device: torch.device) -> dict:
    model.eval()
    total = 0.0
    count = 0
    for offset in range(0, len(records), 64):
        part = records[offset:offset + 64]
        ids, labels, real = batch(part, tokenizer, device)
        with autocast_for(device):
            logits = model(ids, real)
        total += float(F.cross_entropy(
            logits.float().transpose(1, 2), labels,
            ignore_index=-100, reduction="sum",
        ))
        count += int((labels != -100).sum())
    return {"nll": total / count, "scored_response_and_eos_tokens": count}


@torch.no_grad()
def free_generate(model, records: list[dict], tokenizer: Tokenizer,
                  device: torch.device, test_inputs: tuple[int, ...],
                  max_new_tokens: int = 16) -> dict:
    model.eval()
    totals = {"records": len(records), "eos": 0, "parseable": 0,
              "allowed_ast": 0, "passes_finite_tests": 0,
              "symbolically_equivalent": 0, "exact_canonical_text_and_eos": 0}
    examples = []
    for offset in range(0, len(records), 24):
        part = records[offset:offset + 24]
        ids, real, last = prompt_batch(part, tokenizer, device)
        outputs: list[list[int]] = [[] for _ in part]
        finished = torch.zeros(len(part), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            with autocast_for(device):
                logits = model(ids, real)[torch.arange(len(part), device=device), last]
            selected = logits.float().argmax(dim=-1)
            active = ~finished
            for index in range(len(part)):
                if bool(active[index]):
                    outputs[index].append(int(selected[index]))
            continuing = active & (selected != EOS)
            if bool(continuing.any()):
                needed = int((last[continuing] + 1).max())
                if needed >= ids.shape[1]:
                    ids = torch.cat((ids, torch.full(
                        (len(part), needed + 1 - ids.shape[1]), PAD,
                        dtype=torch.long, device=device,
                    )), dim=1)
                    real = torch.cat((real, torch.zeros(
                        (len(part), needed + 1 - real.shape[1]),
                        dtype=torch.bool, device=device,
                    )), dim=1)
                for index in range(len(part)):
                    if bool(continuing[index]):
                        position = int(last[index]) + 1
                        ids[index, position] = selected[index]
                        real[index, position] = True
                        last[index] = position
            finished = finished | (active & (selected == EOS))
            if bool(finished.all()):
                break
        for row, generated in zip(part, outputs):
            has_eos = EOS in generated
            cleaned = generated[:generated.index(EOS)] if has_eos else generated
            text = tokenizer.decode(cleaned).strip()
            checked = check_expression(text, row, test_inputs)
            totals["eos"] += int(has_eos)
            for field in ("parseable", "allowed_ast", "passes_finite_tests",
                          "symbolically_equivalent"):
                totals[field] += int(checked[field])
            totals["exact_canonical_text_and_eos"] += int(
                has_eos and text == row["answer"]
            )
            if len(examples) < 5:
                examples.append({
                    "family": row["family"], "variable": row["variable"],
                    "constant": row["constant"], "expected": row["answer"],
                    "generated": text, "eos": has_eos,
                    "checks": checked, "token_ids": generated,
                })
    totals["max_new_tokens"] = max_new_tokens
    totals["examples"] = examples
    return totals


def probe(out: Path) -> None:
    manifest, manifest_sha = load_manifest()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    device = choose_device()
    model, cfg, identity = start_model(device)
    validation = load_split("validation")
    result = {
        "scope": "C25 full-mix before code-expression SFT; test unopened",
        "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity, "context_limit": cfg.context,
        "validation_teacher_nll": teacher_nll(model, validation, tokenizer, device),
        "validation_free": free_generate(
            model, validation, tokenizer, device, VALIDATION_TEST_INPUTS
        ),
        "checker_sanity": {
            "commuted_add": check_expression(
                "7 + n", {"variable": "n", "family": "add", "constant": 7},
                VALIDATION_TEST_INPUTS
            ),
            "malicious_call_rejected": check_expression(
                "__import__('os').system('echo bad')",
                {"variable": "n", "family": "add", "constant": 7},
                VALIDATION_TEST_INPUTS
            ),
        },
    }
    write_json(out, result)
    print("probe NLL", result["validation_teacher_nll"]["nll"],
          "behavior", result["validation_free"]["passes_finite_tests"])


def train(out_dir: Path, steps: int, learning_rate: float) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _, manifest_sha = load_manifest()
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
    base_free = free_generate(
        model, validation, tokenizer, device, VALIDATION_TEST_INPUTS
    )
    best_key = (base_free["symbolically_equivalent"], -base_nll["nll"])
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"step": 0, "validation_nll": base_nll,
                "validation_free": base_free}]
    first_update = None
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(training), size=32)
        part = [training[int(index)] for index in indices]
        ids, labels, real = batch(part, tokenizer, device)
        model.train()
        with autocast_for(device):
            logits = model(ids, real)
        total = F.cross_entropy(
            logits.float().transpose(1, 2), labels,
            ignore_index=-100, reduction="sum",
        )
        scored = int((labels != -100).sum())
        loss = total / scored
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step == 1:
            first_update = {"batch_records": len(part), "scored_response_tokens": scored,
                            "nll": float(loss.detach()), "gradient_norm": grad_norm}
        if step % 100 == 0:
            current_nll = teacher_nll(model, validation, tokenizer, device)
            current_free = free_generate(
                model, validation, tokenizer, device, VALIDATION_TEST_INPUTS
            )
            history.append({"step": step, "validation_nll": current_nll,
                            "validation_free": current_free})
            key = (current_free["symbolically_equivalent"], -current_nll["nll"])
            if key > best_key:
                best_key = key
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print("step", step, "NLL", round(current_nll["nll"], 4),
                  "AST", current_free["allowed_ast"],
                  "finite", current_free["passes_finite_tests"],
                  "symbolic", current_free["symbolically_equivalent"],
                  flush=True)
    model.load_state_dict(best_state)
    checkpoint = {
        "model": {name: value.detach().cpu().clone()
                  for name, value in model.state_dict().items()},
        "selected_step": best_step,
        "task_manifest_sha256": manifest_sha,
        "start_model_identity": identity,
        "config": cfg.__dict__,
    }
    best_path = out_dir / "best.pt"
    torch.save(checkpoint, best_path)
    report = {
        "scope": "response-only SFT on author-generated short Python expressions from C25 same-model branch; not arbitrary Python or repository repair",
        "task_manifest_sha256": manifest_sha, "start_model_identity": identity,
        "device": str(device), "torch_version": torch.__version__,
        "training": {
            "steps": steps, "batch_size": 32, "learning_rate": learning_rate,
            "selection": "highest validation exact polynomial equivalence, tie by lower response NLL every 100 steps",
            "finite_validation_inputs": VALIDATION_TEST_INPUTS,
        },
        "first_update": first_update, "history": history,
        "selected_step": best_step,
        "selected_validation_nll": teacher_nll(model, validation, tokenizer, device),
        "selected_validation_free": free_generate(
            model, validation, tokenizer, device, VALIDATION_TEST_INPUTS
        ),
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
        raise RuntimeError("training identity changed")
    device = choose_device()
    base, _, identity = start_model(device)
    if identity != report["start_model_identity"]:
        raise RuntimeError("C25 start identity changed")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (state["task_manifest_sha256"] != manifest_sha
            or state["start_model_identity"] != identity
            or state["selected_step"] != report["selected_step"]):
        raise RuntimeError("selected checkpoint changed")
    trained = copy.deepcopy(base)
    trained.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    # The test source is opened only here.
    test = load_split("test")
    paraphrase = [{**row, "prompt": prompt(
        row["family"], row["variable"], row["constant"], paraphrase=True
    )} for row in test]
    new_variable = [{
        "family": family, "variable": "z", "constant": constant,
        "prompt": prompt(family, "z", constant),
        "answer": canonical(family, "z", constant),
    } for family in FAMILIES for constant in (4, 11, 19, 27)]
    results = {}
    for name, model in (("start_c25_full_mix", base),
                        ("selected_code_expression_sft", trained)):
        results[name] = {
            "test_same_template_nll": teacher_nll(model, test, tokenizer, device),
            "test_same_template_free": free_generate(
                model, test, tokenizer, device, HELDOUT_TEST_INPUTS
            ),
            "test_paraphrase_free": free_generate(
                model, paraphrase, tokenizer, device, HELDOUT_TEST_INPUTS
            ),
            "new_variable_z_free": free_generate(
                model, new_variable, tokenizer, device, HELDOUT_TEST_INPUTS
            ),
        }
    output = {
        "scope": "test opened once after validation-selected checkpoint; generated text interpreted only via restricted AST, no exec/eval; paraphrase and z variable fixed beforehand",
        "task_manifest_sha256": manifest_sha,
        "selected_checkpoint_sha256": report["checkpoint_sha256"],
        "selected_step": report["selected_step"],
        "results": results,
    }
    write_json(out, output)
    print("test symbolic", results["start_c25_full_mix"]["test_same_template_free"]["symbolically_equivalent"],
          "to", results["selected_code_expression_sft"]["test_same_template_free"]["symbolically_equivalent"],
          "of", len(test))


def main() -> None:
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "probe", "train", "evaluate"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("work/data/c27_code_expression"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("work/runs/c27_code_expression_trial"))
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    args = parser.parse_args()
    DATA = ROOT / args.data_dir
    if args.mode == "prepare":
        prepare()
    elif args.mode == "probe":
        probe(ROOT / args.out if args.out else
              ROOT / "work/results/c27_code_initial_probe.json")
    elif args.mode == "train":
        train(ROOT / args.run_dir, args.steps, args.learning_rate)
    else:
        evaluate(ROOT / args.run_dir,
                 ROOT / args.out if args.out else
                 ROOT / "work/results/c27_code_first_test.json")


if __name__ == "__main__":
    main()
