"""Train a small explicit copy gate on top of frozen C38 tool-action weights.

The host exposes the token span that came from a tool and its found/not-found
status. A fixed soft positional pointer offers a token from that span at the
same answer ordinal; the learned gate chooses between copying and the LM's
ordinary vocabulary probability. This is extra architecture, not the original
C19 Transformer spontaneously learning reliable arbitrary retrieval.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from c38_tool_data_v3 import tokenized_v3
from c38_tool_train import ROOT, PARENT, TOKENIZER, sha, parent_model, autocast
from transformer_bridge import positional_encoding

DATA = ROOT / "data/c38_tool_trajectories_v3"
ACTION = ROOT / "runs/c38_tool_aligned_copy_2400/best.pt"
KINDS = ("multiply", "title", "missing")


@torch.no_grad()
def hidden_and_logits(model, ids, real, device):
    batch, length = ids.shape
    allowed = torch.ones((length, length), device=device, dtype=torch.bool).tril()[None, None] & real[:, None, None, :]
    with autocast(device):
        x = model.embed(ids) * math.sqrt(model.cfg.width)
        x = x + positional_encoding(length, model.cfg.width, device)[None]
        for block in model.blocks:
            x = block(x, allowed)
        logits = model.output(x)
    return x.float(), logits.float()


class CopyGate(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.use_copy = nn.Linear(width + 1, 1)
        nn.init.zeros_(self.use_copy.weight)
        nn.init.constant_(self.use_copy.bias, -2.0)
        self.raw_distance_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, hidden, found, ordinals, observed_ids, observed_valid, targets, lm_target_probability):
        gate = torch.sigmoid(self.use_copy(torch.cat((hidden, found[:, None]), dim=-1))).squeeze(-1)
        distance = (torch.arange(observed_ids.shape[1], device=hidden.device)[None] - ordinals[:, None]).abs()
        scores = -F.softplus(self.raw_distance_scale) * distance.float()
        scores = scores.masked_fill(~observed_valid, -1e9)
        attention = scores.softmax(dim=-1)
        copy_target_probability = (attention * ((observed_ids == targets[:, None]) & observed_valid)).sum(dim=-1)
        probability = (1 - gate) * lm_target_probability + gate * copy_target_probability
        return probability.clamp_min(1e-12), gate, copy_target_probability


def load_copy_gate(path: Path, device, action_checkpoint: Path = ACTION):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["action_checkpoint_sha256"] != sha(action_checkpoint) or state["data_manifest_sha256"] != sha(DATA / "manifest.json"):
        raise RuntimeError("copy gate is not bound to this frozen C38 action model and v3 data")
    gate = CopyGate().to(device)
    gate.load_state_dict(state["copy_gate"])
    gate.eval()
    return gate, state


@torch.inference_mode()
def copy_line(model, gate, tok, prefix_ids: list[int], observation_value_ids: list[int], found: bool, device, max_new: int = 32):
    ids = list(prefix_ids)
    generated = []
    stop = "limit"
    source = torch.tensor(observation_value_ids, dtype=torch.long, device=device)
    if len(source) == 0:
        raise ValueError("tool observation has no value tokens")
    scale = F.softplus(gate.raw_distance_scale)
    for ordinal in range(max_new):
        window = torch.tensor(ids[-model.cfg.context:], dtype=torch.long, device=device)[None]
        hidden, logits = hidden_and_logits(model, window, torch.ones_like(window, dtype=torch.bool), device)
        feature = torch.cat((hidden[0, -1], torch.tensor([float(found)], device=device)))
        mix = gate.use_copy(feature).sigmoid().squeeze()
        distances = (torch.arange(len(source), device=device) - ordinal).abs().float()
        attention = (-scale * distances).softmax(dim=0)
        copy = torch.zeros(model.cfg.vocab, device=device)
        copy.scatter_add_(0, source, attention)
        probabilities = (1 - mix) * logits[0, -1].softmax(dim=-1) + mix * copy
        probabilities[[0, 1]] = -1
        next_id = int(probabilities.argmax())
        if next_id == 2:
            stop = "eos"
            break
        generated.append(next_id)
        ids.append(next_id)
        if "\n" in tok.decode(generated, skip_special_tokens=False):
            stop = "newline"
            break
    return {"token_ids": generated, "text": tok.decode(generated, skip_special_tokens=False), "stop": stop,
            "copy_source_token_count": len(observation_value_ids), "found_status": bool(found)}


def load_action_model(device, action_checkpoint: Path = ACTION):
    model, cfg, _ = parent_model(device)
    action = torch.load(action_checkpoint, map_location="cpu", weights_only=True)
    if action["data_manifest_sha256"] != sha(DATA / "manifest.json") or action["parent_checkpoint_sha256"] != sha(PARENT):
        raise RuntimeError("frozen action checkpoint not bound to C38 v3 data")
    model.load_state_dict(action["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, action


@torch.no_grad()
def features_for_split(name, model, tokenizer, device):
    rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["kind"] in KINDS]
    enc = [tokenized_v3(tokenizer, row) for row in rows]
    inputs = np.array([row["input_ids"] for row in enc], dtype=np.int64)
    labels = np.array([row["labels"] for row in enc], dtype=np.int64)
    real = np.array([row["attention_mask"] for row in enc], dtype=np.bool_)
    ordinals = np.array([row["copy_final_target_ordinals"] for row in enc], dtype=np.int64)
    selected = []
    max_source = max(row["copy_observation_value_length"] for row in enc)
    for start in range(0, len(rows), 32):
        end = min(start + 32, len(rows))
        x = torch.as_tensor(inputs[start:end], dtype=torch.long, device=device)
        m = torch.as_tensor(real[start:end], dtype=torch.bool, device=device)
        hidden, logits = hidden_and_logits(model, x, m, device)
        y = torch.as_tensor(labels[start:end], dtype=torch.long, device=device)
        gather = y.clamp_min(0)
        lm_prob = logits.softmax(dim=-1).gather(-1, gather[..., None]).squeeze(-1)
        for local, original_index in enumerate(range(start, end)):
            row, e = rows[original_index], enc[original_index]
            obs_start = e["copy_observation_value_start"]
            obs_length = e["copy_observation_value_length"]
            obs = inputs[original_index, obs_start:obs_start + obs_length]
            padded_obs = np.pad(obs, (0, max_source - len(obs)), constant_values=0)
            valid = [True] * len(obs) + [False] * (max_source - len(obs))
            for position, ordinal in enumerate(ordinals[original_index]):
                if ordinal < 0:
                    continue
                selected.append((hidden[local, position].cpu().numpy(),
                                 row["kind"] != "missing", ordinal, padded_obs.copy(), valid,
                                 labels[original_index, position], float(lm_prob[local, position]), row["kind"]))
    if not selected:
        raise RuntimeError("no final assistant targets")
    data = {
        "hidden": torch.tensor(np.stack([v[0] for v in selected]), dtype=torch.float32, device=device),
        "found": torch.tensor([v[1] for v in selected], dtype=torch.float32, device=device),
        "ordinal": torch.tensor([v[2] for v in selected], dtype=torch.long, device=device),
        "observed_ids": torch.tensor(np.stack([v[3] for v in selected]), dtype=torch.long, device=device),
        "observed_valid": torch.tensor(np.array([v[4] for v in selected]), dtype=torch.bool, device=device),
        "target": torch.tensor([v[5] for v in selected], dtype=torch.long, device=device),
        "lm_probability": torch.tensor([v[6] for v in selected], dtype=torch.float32, device=device),
        "kind": [v[7] for v in selected],
        "rows": len(rows),
    }
    return data


def indexed(data, indices):
    return [data[name][indices] for name in ("hidden", "found", "ordinal", "observed_ids", "observed_valid", "target", "lm_probability")]


@torch.no_grad()
def evaluate(gate, data):
    gate.eval()
    result = {}
    for kind in KINDS:
        index = torch.tensor([i for i, value in enumerate(data["kind"]) if value == kind], device=data["hidden"].device)
        probability, mix, copy = gate(*indexed(data, index))
        result[kind] = {"target_nll": float(-probability.log().mean()),
                        "targets": len(index), "mean_copy_gate": float(mix.mean()),
                        "copy_target_probability_mean": float(copy.mean()),
                        "plain_lm_target_nll": float(-data["lm_probability"][index].clamp_min(1e-12).log().mean())}
    return {"macro_target_nll": sum(v["target_nll"] for v in result.values()) / len(result), "by_kind": result}


def train(out_dir, steps, lr, validate_every, action_checkpoint):
    if out_dir.exists():
        raise RuntimeError("output exists")
    if not (steps > 0 and 0 < lr < 0.1 and validate_every > 0):
        raise ValueError("invalid plan")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 3
    for name in ("train.jsonl", "validation.jsonl"):
        if sha(DATA / name) != manifest["files"][name]:
            raise RuntimeError(name + " changed")
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, action = load_action_model(device, action_checkpoint)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    started = time.perf_counter()
    train_data = features_for_split("train", model, tokenizer, device)
    val_data = features_for_split("validation", model, tokenizer, device)
    gate = CopyGate(model.cfg.width).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=0.0)
    pools = {kind: np.array([i for i, v in enumerate(train_data["kind"]) if v == kind]) for kind in KINDS}
    rng = np.random.default_rng(20260925)
    start_val = evaluate(gate, val_data)
    best, best_step = start_val["macro_target_nll"], 0
    best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    history = [{"step": 0, "validation": start_val}]
    for step in range(1, steps + 1):
        gate.train()
        positions = np.concatenate([rng.choice(pools[kind], 32, replace=True) for kind in KINDS])
        rng.shuffle(positions)
        index = torch.tensor(positions, device=device)
        optimizer.zero_grad(set_to_none=True)
        probability, _, _ = gate(*indexed(train_data, index))
        loss = -probability.log().mean()
        loss.backward()
        optimizer.step()
        if step % validate_every == 0 or step == steps:
            measure = evaluate(gate, val_data)
            history.append({"step": step, "validation": measure})
            print(json.dumps({"step": step, "macro": measure["macro_target_nll"], "by_kind": {k: round(v["target_nll"], 4) for k, v in measure["by_kind"].items()}, "gate": {k: round(v["mean_copy_gate"], 3) for k, v in measure["by_kind"].items()}}, ensure_ascii=False), flush=True)
            if measure["macro_target_nll"] < best:
                best, best_step = measure["macro_target_nll"], step
                best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    out_dir.mkdir(parents=True)
    checkpoint = out_dir / "copy_gate.pt"
    torch.save({"copy_gate": best_state, "selected_step": best_step,
                "action_checkpoint_sha256": sha(action_checkpoint), "data_manifest_sha256": sha(DATA / "manifest.json"),
                "selection_macro_target_nll": best, "copy_rule": "softmax(-softplus(scale)*absolute(source_ordinal-answer_ordinal)); gate mixes this with frozen LM vocabulary"}, checkpoint)
    report = {"task": "C38 explicit source-position copy gate with frozen v3 action LM; no new tool decision training and no test read",
              "command": f"python work/code/c38_copy_gate.py --out-dir {out_dir.relative_to(ROOT.parent)} --steps {steps} --lr {lr} --validate-every {validate_every} --action-checkpoint {action_checkpoint.relative_to(ROOT.parent)}",
              "source_manifest_sha256": sha(DATA / "manifest.json"), "action_checkpoint_sha256": sha(action_checkpoint),
              "train_rows_with_tool_observation": train_data["rows"], "validation_rows_with_tool_observation": val_data["rows"],
              "train_final_targets": len(train_data["kind"]), "validation_final_targets": len(val_data["kind"]),
              "parameters_added": sum(p.numel() for p in gate.parameters()),
              "host_supplied_copy_information": "exact tool observation value token span, answer token ordinal and found/not-found flag; selected action still comes from frozen C38 v3 LM",
              "history": history, "selected_step": best_step, "selected_validation_macro_target_nll": best,
              "selected_gate_distance_scale": float(F.softplus(best_state["raw_distance_scale"])),
              "checkpoint": str(checkpoint.relative_to(ROOT.parent)).replace("\\", "/"), "checkpoint_sha256": sha(checkpoint),
              "elapsed_seconds": time.perf_counter() - started, "device": str(device), "torch": torch.__version__, "test_opened": False}
    (out_dir / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_step": best_step, "validation_macro_final_nll": best, "distance_scale": report["selected_gate_distance_scale"], "checkpoint_sha256": report["checkpoint_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--validate-every", type=int, default=20)
    p.add_argument("--action-checkpoint", type=Path, default=ACTION)
    args = p.parse_args()
    out = args.out_dir.resolve()
    if not out.is_relative_to(ROOT / "runs"):
        raise ValueError("run directory outside work/runs")
    action_checkpoint = args.action_checkpoint.resolve()
    if not action_checkpoint.is_relative_to(ROOT / "runs"):
        raise ValueError("action checkpoint outside work/runs")
    train(out, args.steps, args.lr, args.validate_every, action_checkpoint)
