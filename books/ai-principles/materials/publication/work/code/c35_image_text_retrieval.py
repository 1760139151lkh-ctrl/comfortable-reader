"""Small jointly trained COCO image/text matcher; no external model weights."""
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "c35_multimodal_local"


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.gru = nn.GRU(128, 192, batch_first=True)
        self.project = nn.Linear(192, out_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.project(hidden[-1])


class PairModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.image = ImageEncoder()
        self.text = TextEncoder(vocab_size)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward(self, image: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.image(image), dim=-1)
        v = F.normalize(self.text(tokens, lengths), dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * (u @ v.T)


def batch_tokens(sequences: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([max(1, len(x)) for x in sequences], dtype=torch.long, device=device)
    width = int(lengths.max())
    out = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        if seq:
            out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        else:
            out[i, 0] = 3
    return out, lengths


@torch.inference_mode()
def retrieval_metrics(model: PairModel, images: np.ndarray, rows: list[dict], indices: list[int], device: torch.device) -> dict:
    model.eval()
    img_vec, txt_vec = [], []
    for start in range(0, len(indices), 64):
        ids = indices[start:start + 64]
        x = torch.from_numpy(np.asarray(images[ids], dtype=np.float32).copy()).to(device) / 255.0
        seq = [rows[i]["caption_tokens"][0] for i in ids]
        tok, lengths = batch_tokens(seq, device)
        img_vec.append(F.normalize(model.image(x), dim=-1).cpu())
        txt_vec.append(F.normalize(model.text(tok, lengths), dim=-1).cpu())
    sim = torch.cat(img_vec) @ torch.cat(txt_vec).T
    ranks = {}
    for label, scores in (("image_to_text", sim), ("text_to_image", sim.T)):
        order = scores.argsort(dim=1, descending=True)
        matches = (order == torch.arange(len(indices))[:, None]).nonzero()
        rank = matches[:, 1] + 1
        ranks[label] = {
            "recall_at_1": float((rank <= 1).float().mean()),
            "recall_at_5": float((rank <= 5).float().mean()),
            "recall_at_10": float((rank <= 10).float().mean()),
            "median_rank": float(rank.float().median()),
            "mean_rank": float(rank.float().mean()),
        }
    ranks["mean_recall_at_1"] = (ranks["image_to_text"]["recall_at_1"] + ranks["text_to_image"]["recall_at_1"]) / 2
    ranks["candidate_images"] = len(indices)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["true", "wrong"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("nonempty run dir; choose another --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = json.loads((DATA / "image_records.json").read_text(encoding="utf-8"))
    images = np.load(DATA / f"images_{manifest['image_side']}_chw_uint8.npy", mmap_mode="r")
    vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    train_ids = np.array([i for i, r in enumerate(rows) if r["split"] == "train"], dtype=np.int64)
    val_ids = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    if len(train_ids) != 4000 or len(val_ids) != 500:
        raise RuntimeError("unexpected split")
    model = PairModel(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.01)
    history: list[dict] = []
    best = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train_ids)
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(train_ids), args.batch_size):
            ids = train_ids[start:start + args.batch_size].tolist()
            if len(ids) < 2:
                continue
            # Exactly one unique image per batch. Five captions of one photo
            # would otherwise be false negatives of each other.
            selected = [rows[i]["caption_tokens"][(epoch + int(i)) % len(rows[i]["caption_tokens"])] for i in ids]
            if args.mode == "wrong":
                selected = selected[1:] + selected[:1]
            x = torch.from_numpy(np.asarray(images[ids], dtype=np.float32).copy()).to(device) / 255.0
            tok, lengths = batch_tokens(selected, device)
            logits = model(x, tok, lengths)
            target = torch.arange(len(ids), device=device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            steps += 1
        val = retrieval_metrics(model, images, rows, val_ids, device)
        item = {"epoch": epoch, "train_pair_loss": epoch_loss / steps, "validation": val}
        history.append(item)
        metric = val["mean_recall_at_1"]
        if metric > best:
            best = metric
            torch.save({"model": model.state_dict(), "vocab_size": len(vocab), "epoch": epoch, "mode": args.mode, "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}, out / "best.pt")
        print(json.dumps({"epoch": epoch, "train_loss": item["train_pair_loss"], "val_r1": metric, "best": best}), flush=True)
    result = {
        "mode": args.mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_images": 4000,
        "validation_images": 500,
        "vocab_size": len(vocab),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "elapsed_seconds": time.time() - start_time,
        "selection": "highest validation mean recall@1, ties earlier epoch; test images never loaded here",
        "image_text_source": "COCO2017 official validation repartitioned by image ID for author-internal lab",
        "history": history,
    }
    (out / "train.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
