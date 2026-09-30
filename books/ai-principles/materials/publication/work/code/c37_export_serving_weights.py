"""Export exactly the C19 trained model weights, excluding optimizer/RNG state."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from train_causal_wikitext import CausalLM, TrainConfig

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "causal_lm_best.pt"
SUMMARY = ROOT / "results" / "causal_lm_train.json"
DATA = ROOT / "data" / "wikitext2_causal"
OUT = ROOT / "data" / "c37_serving"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "c19_model_only.pt"
    receipt = OUT / "export_receipt.json"
    if destination.exists() or receipt.exists():
        raise RuntimeError("serving export exists; preserve it or choose a new version")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    source_sha = sha(SOURCE)
    if source_sha != summary["best_checkpoint_sha256"]:
        raise RuntimeError("C19 source checkpoint differs from training receipt")
    state = torch.load(SOURCE, map_location="cpu", weights_only=True)
    manifest_sha = sha(DATA / "manifest.json")
    if state["c18_manifest_sha256"] != manifest_sha or state["config"] != asdict(TrainConfig()):
        raise RuntimeError("C19 checkpoint/config/data identity mismatch")
    model_state = {key: tensor.detach().cpu().clone() for key, tensor in state["model"].items()}
    serving = {"model": model_state, "config": state["config"], "c18_manifest_sha256": manifest_sha, "tokenizer_sha256": sha(DATA / "tokenizer.json"), "source_checkpoint_sha256": source_sha, "selected_epoch": state["best_epoch"]}
    torch.save(serving, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    if list(reloaded["model"]) != list(model_state) or any(not torch.equal(reloaded["model"][key], value) for key, value in model_state.items()):
        raise RuntimeError("weight-only export altered a tensor")
    before = CausalLM(TrainConfig()).eval()
    after = CausalLM(TrainConfig()).eval()
    before.load_state_dict(state["model"])
    after.load_state_dict(reloaded["model"])
    blocks = np.load(DATA / "train_blocks.npz")
    ids = torch.as_tensor(blocks["input_ids"][0, :16], dtype=torch.long)[None]
    real = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        max_logit_difference = float((before(ids, real) - after(ids, real)).abs().max())
    if max_logit_difference != 0:
        raise RuntimeError("serving weights changed C19 logits")
    result = {"identity": "C19 randomly initialized then four-epoch WikiText-trained 4,930,304-parameter causal LM; no new learning during export", "original_checkpoint": str(SOURCE), "original_bytes": SOURCE.stat().st_size, "original_sha256": source_sha, "serving_file": str(destination), "serving_bytes": destination.stat().st_size, "serving_sha256": sha(destination), "dropped": "AdamW states, epoch cursor and RNG; only fixed model/config/data identity/tokenizer identity retained", "state_tensors_bitwise_equal": len(model_state), "same_train_block_16_position_logits_max_difference": max_logit_difference, "tokenizer_sha256": serving["tokenizer_sha256"], "c18_manifest_sha256": manifest_sha}
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
