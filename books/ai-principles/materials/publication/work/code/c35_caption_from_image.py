"""Train a small caption decoder on frozen *locally trained* visual features."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from c35_image_text_retrieval import DATA, ImageEncoder


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, mode: str):
        super().__init__()
        self.mode = mode
        self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.image_to_hidden = nn.Linear(128, 256)
        self.gru = nn.GRU(128, 256, batch_first=True)
        self.output = nn.Linear(256, vocab_size)

    def forward(self, visual: torch.Tensor, tokens_in: torch.Tensor) -> torch.Tensor:
        if self.mode == "text_only":
            visual = torch.zeros_like(visual)
        hidden = torch.tanh(self.image_to_hidden(visual)).unsqueeze(0)
        sequence, _ = self.gru(self.embed(tokens_in), hidden)
        return self.output(sequence)

    @torch.inference_mode()
    def generate(self, visual: torch.Tensor, max_tokens: int = 30) -> list[int]:
        self.eval()
        if self.mode == "text_only":
            visual = torch.zeros_like(visual)
        hidden = torch.tanh(self.image_to_hidden(visual)).unsqueeze(0)
        current = torch.ones((len(visual), 1), dtype=torch.long, device=visual.device)  # <bos>
        sequences = [[] for _ in range(len(visual))]
        ended = [False] * len(visual)
        for _ in range(max_tokens):
            state, hidden = self.gru(self.embed(current), hidden)
            next_token = self.output(state[:, -1]).argmax(dim=-1)
            current = next_token[:, None]
            for i, item in enumerate(next_token.tolist()):
                if not ended[i]:
                    if item == 2:
                        ended[i] = True
                    else:
                        sequences[i].append(item)
            if all(ended):
                break
        return sequences


def caption_batch(rows: list[dict], item_indices: list[tuple[int, int]], device: torch.device):
    seqs = [rows[image_idx]["caption_tokens"][caption_idx] for image_idx, caption_idx in item_indices]
    width = max(len(s) for s in seqs) + 1
    inp = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
    target = torch.zeros_like(inp)
    for i, seq in enumerate(seqs):
        inp[i, 0] = 1
        inp[i, 1:len(seq) + 1] = torch.tensor(seq, device=device)
        target[i, :len(seq)] = torch.tensor(seq, device=device)
        target[i, len(seq)] = 2
    return inp, target


@torch.inference_mode()
def make_features(encoder: ImageEncoder, images: np.ndarray, indices: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    encoder.eval()
    feature = {}
    for start in range(0, len(indices), 128):
        ids = indices[start:start + 128]
        x = torch.from_numpy(np.asarray(images[ids], dtype=np.float32).copy()).to(device) / 255.0
        vectors = encoder(x).cpu()
        for i, vec in zip(ids, vectors):
            feature[i] = vec
    return feature


@torch.inference_mode()
def evaluate(model: CaptionDecoder, rows: list[dict], features: dict[int, torch.Tensor], items: list[tuple[int, int]], device: torch.device, shuffled: bool = False):
    model.eval()
    loss_sum, token_count = 0.0, 0
    all_indices = sorted({i for i, _ in items})
    replacement = {image_idx: all_indices[(j + 31) % len(all_indices)] for j, image_idx in enumerate(all_indices)}
    for start in range(0, len(items), 128):
        batch = items[start:start + 128]
        visual = torch.stack([features[replacement[i] if shuffled else i] for i, _ in batch]).to(device)
        inp, target = caption_batch(rows, batch, device)
        logits = model(visual, inp)
        loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0, reduction="sum"))
        token_count += int((target != 0).sum())
    return {"captions": len(items), "target_tokens": token_count, "nll_per_target_token": loss_sum / token_count}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image_conditioned", "text_only"], required=True)
    p.add_argument("--retrieval-ckpt", type=Path, default=Path("work/runs/c35_retrieval_true30/best.pt"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise RuntimeError("nonempty run directory")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = json.loads((DATA / "image_records.json").read_text(encoding="utf-8"))
    vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    images = np.load(DATA / f"images_{manifest['image_side']}_chw_uint8.npy", mmap_mode="r")
    ckpt = torch.load(a.retrieval_ckpt, map_location="cpu", weights_only=True)
    image_state = {k.removeprefix("image."): v for k, v in ckpt["model"].items() if k.startswith("image.")}
    encoder = ImageEncoder().to(device)
    encoder.load_state_dict(image_state)
    train_image_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val_image_ids = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    features = make_features(encoder, images, train_image_ids + val_image_ids, device)
    train_items = [(i, c) for i in train_image_ids for c in range(len(rows[i]["captions"]))]
    val_items = [(i, c) for i in val_image_ids for c in range(len(rows[i]["captions"]))]
    model = CaptionDecoder(len(vocab), a.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    history, best = [], float("inf")
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        random.shuffle(train_items)
        loss_sum, tokens = 0.0, 0
        for start in range(0, len(train_items), 128):
            batch = train_items[start:start + 128]
            visual = torch.stack([features[i] for i, _ in batch]).to(device)
            inp, target = caption_batch(rows, batch, device)
            logits = model(visual, inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0, reduction="sum")
            n = int((target != 0).sum())
            optimizer.zero_grad(set_to_none=True)
            (loss / n).backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            tokens += n
        val = evaluate(model, rows, features, val_items, device)
        history.append({"epoch": epoch, "train_nll_per_target_token": loss_sum / tokens, "validation": val})
        if val["nll_per_target_token"] < best:
            best = val["nll_per_target_token"]
            torch.save({"model": model.state_dict(), "mode": a.mode, "epoch": epoch, "vocab_size": len(vocab), "retrieval_ckpt_sha256": hashlib.sha256(a.retrieval_ckpt.read_bytes()).hexdigest(), "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}, a.out_dir / "best.pt")
        print(json.dumps({"epoch": epoch, "train_nll": loss_sum / tokens, "val_nll": val["nll_per_target_token"], "best": best}), flush=True)
    report = {"mode": a.mode, "frozen_visual_checkpoint": str(a.retrieval_ckpt), "frozen_visual_selected_epoch": ckpt["epoch"], "train_captions": len(train_items), "validation_captions": len(val_items), "test_used": False, "decoder_parameters": sum(p.numel() for p in model.parameters()), "device": str(device), "elapsed_seconds": time.time() - t0, "selection": "minimum internal validation teacher-forced token NLL; test untouched", "history": history}
    (a.out_dir / "train.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
