"""C38: free model actions, real read-only host tools, and source-separated scores."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from c38_tool_data import TOOLS, encode
from c38_tool_train import DATA, PARENT, ROOT, TOKENIZER, KINDS, parent_model, sha, autocast
from c38_copy_gate import copy_line, load_copy_gate

CALL_MUL = re.compile(r"call mul\(([0-9]{1,2}),([0-9]{1,2})\)")
CALL_TITLE = re.compile(r"call title\(([0-9]{1,3})\)")


@torch.inference_mode()
def line(model, tok, prefix_ids: list[int], device, max_new: int = 32):
    ids = list(prefix_ids)
    generated = []
    stop = "limit"
    model.eval()
    for _ in range(max_new):
        window = torch.tensor(ids[-model.cfg.context:], device=device, dtype=torch.long)[None]
        with autocast(device):
            logits = model(window, torch.ones_like(window, dtype=torch.bool))[0, -1].float()
        logits[[0, 1]] = float("-inf")
        next_id = int(logits.argmax())
        if next_id == 2:
            stop = "eos"
            break
        generated.append(next_id)
        ids.append(next_id)
        text = tok.decode(generated, skip_special_tokens=False)
        if "\n" in text:
            stop = "newline"
            break
    return {"token_ids": generated, "text": tok.decode(generated, skip_special_tokens=False), "stop": stop}


def execute(text: str, catalog: dict[str, str]):
    if not text.endswith("\n"):
        return {"valid": False, "error": "no newline terminator"}
    command = text[:-1].strip()
    mul = CALL_MUL.fullmatch(command)
    if mul:
        a, b = map(int, mul.groups())
        if not (10 <= a <= 99 and 10 <= b <= 99):
            return {"valid": False, "error": "multiply arguments outside 10..99"}
        return {"valid": True, "name": "mul", "args": [a, b], "observation": str(a * b)}
    title = CALL_TITLE.fullmatch(command)
    if title:
        index = int(title.group(1))
        return {"valid": True, "name": "title", "args": [index], "observation": catalog.get(str(index), "NOT_FOUND")}
    return {"valid": False, "error": "unknown name or malformed call"}


def expected_name(row):
    return "mul" if row["kind"] == "multiply" else "title" if row["kind"] in ("title", "missing") else None


def expected_args(row):
    return [row["a"], row["b"]] if row["kind"] == "multiply" else [row["article_id"]] if row["kind"] in ("title", "missing") else None


@torch.inference_mode()
def evaluate_one(row, model, tok, device, catalog, format_version, gate=None):
    first_prefix = TOOLS + "Q: " + row["question"] + ("\nA:" if format_version == 3 else "\nA: ")
    prefix = [1] + encode(tok, first_prefix)
    first = line(model, tok, prefix, device)
    is_call = first["text"].strip().startswith("call ")
    wants_call = row["kind"] != "direct"
    route_correct = is_call == wants_call
    result = {"kind": row["kind"], "question": row["question"], "expected_action": row["action"],
              "expected_answer": row["answer"], "first": first, "route_correct": route_correct,
              "call_parse_valid": False, "name_correct": False, "arguments_correct": False,
              "answer": None, "answer_correct": False, "task_success": False,
              "observation": None}
    if not is_call:
        result["answer"] = first["text"].strip()
        result["answer_correct"] = result["answer"] == row["answer"]
        result["task_success"] = not wants_call and result["answer_correct"] and first["stop"] == "newline"
        return result
    tool = execute(first["text"], catalog)
    result["call_parse_valid"] = tool["valid"]
    if not tool["valid"]:
        result["tool_error"] = tool["error"]
        return result
    result["name_correct"] = tool["name"] == expected_name(row)
    result["arguments_correct"] = result["name_correct"] and tool["args"] == expected_args(row)
    result["observation"] = tool["observation"]
    # The assistant produced first-token IDs are retained as generated, not
    # re-tokenized into a reference call. The host supplies its own observation.
    observation_prefix = "T: " + tool["observation"] + ("\nA:" if format_version == 3 else "\nA: ")
    after = prefix + first["token_ids"] + encode(tok, observation_prefix)
    if gate is None:
        final = line(model, tok, after, device)
    else:
        observation_value_ids = encode(tok, observation_prefix)[2:-2]
        final = copy_line(model, gate, tok, after, observation_value_ids, tool["observation"] != "NOT_FOUND", device)
    result["final"] = final
    result["answer"] = final["text"].strip()
    result["answer_correct"] = result["answer"] == row["answer"]
    result["task_success"] = wants_call and result["arguments_correct"] and result["answer_correct"] and final["stop"] == "newline"
    result["observation_followed"] = (result["answer"] == tool["observation"] if tool["observation"] != "NOT_FOUND" else result["answer"] == "not found")
    return result


def load_model(kind, checkpoint, device, data_dir):
    model, cfg, parent = parent_model(device)
    if kind == "trained":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["config"] != vars(cfg) or state["parent_checkpoint_sha256"] != sha(PARENT) or state["data_manifest_sha256"] != sha(data_dir / "manifest.json"):
            raise RuntimeError("trained model/data/parent identity mismatch")
        model.load_state_dict(state["model"])
        step = state["selected_step"]
        model_sha = sha(checkpoint)
    else:
        step = 0
        model_sha = sha(PARENT)
    model.eval()
    return model, step, model_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--model-kind", choices=("trained", "parent"), default="trained")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/c38_tool_action_sft_1800/best.pt")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--copy-gate", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise RuntimeError("result already exists")
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir.resolve()
    if not data_dir.is_relative_to(ROOT / "data"):
        raise ValueError("data directory outside work/data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    format_version = manifest.get("format_version", 1)
    if format_version not in (1, 3):
        raise RuntimeError("unknown C38 prompt format")
    if manifest["tokenizer_sha256"] != sha(TOKENIZER) or manifest["parent_c25_full_mix_checkpoint_sha256"] != sha(PARENT):
        raise RuntimeError("source drift")
    model, selected_step, model_sha = load_model(args.model_kind, args.checkpoint, device, data_dir)
    gate = None
    gate_sha = None
    if args.copy_gate is not None:
        if args.model_kind != "trained" or format_version != 3:
            raise ValueError("copy gate requires the trained v3 action model")
        gate, _ = load_copy_gate(args.copy_gate, device, args.checkpoint)
        gate_sha = sha(args.copy_gate)
    tok = Tokenizer.from_file(str(TOKENIZER))
    name = args.split + ".jsonl"
    if sha(data_dir / name) != manifest["files"][name] or sha(data_dir / "title_catalog.json") != manifest["files"]["title_catalog.json"]:
        raise RuntimeError("split/catalog changed")
    # For a test run, this is the first model-process read of test.jsonl.
    rows = [json.loads(line) for line in (data_dir / name).read_text(encoding="utf-8").splitlines()]
    catalog = json.loads((data_dir / "title_catalog.json").read_text(encoding="utf-8"))
    traces = [evaluate_one(row, model, tok, device, catalog, format_version, gate) for row in rows]
    by_kind = {}
    for kind in KINDS:
        subset = [entry for entry in traces if entry["kind"] == kind]
        n = len(subset)
        by_kind[kind] = {"rows": n,
                         "route_correct": sum(entry["route_correct"] for entry in subset),
                         "call_parse_valid": sum(entry["call_parse_valid"] for entry in subset),
                         "name_correct": sum(entry["name_correct"] for entry in subset),
                         "arguments_correct": sum(entry["arguments_correct"] for entry in subset),
                         "answer_correct": sum(entry["answer_correct"] for entry in subset),
                         "task_success": sum(entry["task_success"] for entry in subset),
                         "observation_followed": sum(entry.get("observation_followed", False) for entry in subset)}
    report = {
        "scope": "one free assistant line, optional actual read-only tool execution, then one free final line; no constrained action grammar or host correction of model choice",
        "split": args.split, "split_sha256": sha(data_dir / name), "data_manifest_sha256": sha(data_dir / "manifest.json"), "format_version": format_version, "model_kind": args.model_kind,
        "model_checkpoint_sha256": model_sha, "copy_gate_checkpoint_sha256": gate_sha, "parent_checkpoint_sha256": sha(PARENT),
        "selected_training_step": selected_step, "device": str(device), "torch": torch.__version__,
        "rows": len(rows), "by_kind": by_kind,
        "overall_task_success": sum(entry["task_success"] for entry in traces),
        "overall_route_correct": sum(entry["route_correct"] for entry in traces),
        "traces": traces,
        "limitations": "fixed English templates and at most one call; title catalog uses C18 train article titles, some words seen in parent pretraining; not API-Bank/Tau-bench or multi-tool planning",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "model_kind": args.model_kind, "overall_task_success": report["overall_task_success"], "rows": len(rows), "by_kind": {k: {field: v[field] for field in ("rows", "route_correct", "arguments_correct", "task_success")} for k, v in by_kind.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
