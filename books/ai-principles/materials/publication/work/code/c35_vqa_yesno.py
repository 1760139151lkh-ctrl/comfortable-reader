"""VQA v2 yes/no: identical question head with and without learned image features."""
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

from c35_image_text_retrieval import DATA, ImageEncoder, batch_tokens


class YesNoModel(nn.Module):
    def __init__(self, vocab_size: int, image_state: dict, mode: str):
        super().__init__()
        self.mode = mode
        self.image = ImageEncoder()
        self.image.load_state_dict(image_state)
        for param in self.image.parameters():
            param.requires_grad_(False)
        self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.question = nn.GRU(128, 192, batch_first=True)
        self.head = nn.Sequential(nn.Linear(192 + 128 + 128, 192), nn.ReLU(), nn.Linear(192, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.image.eval()  # frozen BatchNorm statistics also stay frozen
        return self

    def forward(self, pixels: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.question(packed)
        q = h[-1]
        if self.mode == "question_only":
            v = torch.zeros((len(q), 128), device=q.device, dtype=q.dtype)
        else:
            with torch.no_grad():
                v = self.image(pixels)
        return self.head(torch.cat([q, v, q[:, :128] * v], dim=-1)).squeeze(-1)


def make_batch(records: list[dict], images: np.ndarray, by_image_id: dict[int, int], indices: list[int], device: torch.device):
    batch = [records[i] for i in indices]
    pixels = np.asarray(images[[by_image_id[r["image_id"]] for r in batch]], dtype=np.float32).copy()
    x = torch.from_numpy(pixels).to(device) / 255.0
    tok, lengths = batch_tokens([r["question_tokens"] for r in batch], device)
    labels = torch.tensor([1.0 if r["answer"] == "yes" else 0.0 for r in batch], device=device)
    return x, tok, lengths, labels


@torch.inference_mode()
def evaluate(model: YesNoModel, records: list[dict], images: np.ndarray, by_image_id: dict[int, int], indices: list[int], device: torch.device) -> dict:
    model.eval()
    nll, correct, total = 0.0, 0, 0
    for start in range(0, len(indices), 128):
        x, tok, lengths, labels = make_batch(records, images, by_image_id, indices[start:start + 128], device)
        logits = model(x, tok, lengths)
        nll += float(F.binary_cross_entropy_with_logits(logits, labels, reduction="sum"))
        correct += int(((logits >= 0).float() == labels).sum())
        total += len(labels)
    return {"questions": total, "nll_per_question": nll / total, "accuracy": correct / total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image_question", "question_only"], required=True)
    parser.add_argument("--retrieval-ckpt", type=Path, default=Path("work/runs/c35_retrieval_true30/best.pt"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("nonempty run dir")
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
    questions = json.loads((DATA / "vqa_records.json").read_text(encoding="utf-8"))
    records = [q for q in questions if q["answer_type"] == "yes/no" and q["answer"] in ("yes", "no")]
    by_image_id = {r["image_id"]: i for i, r in enumerate(rows)}
    train_ids = [i for i, r in enumerate(records) if r["split"] == "train"]
    val_ids = [i for i, r in enumerate(records) if r["split"] == "validation"]
    if len(train_ids) != 7947 or len(val_ids) != 1009:
        raise RuntimeError("split changed")
    images = np.load(DATA / f"images_{manifest['image_side']}_chw_uint8.npy", mmap_mode="r")
    vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    retrieval_ckpt = torch.load(args.retrieval_ckpt, map_location="cpu", weights_only=True)
    image_state = {k.removeprefix("image."): v for k, v in retrieval_ckpt["model"].items() if k.startswith("image.")}
    model = YesNoModel(len(vocab), image_state, args.mode).to(device)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001, weight_decay=0.01)
    history, best = [], float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_ids)
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(train_ids), 128):
            x, tok, lengths, y = make_batch(records, images, by_image_id, train_ids[start:start + 128], device)
            logits = model(x, tok, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            steps += 1
        val = evaluate(model, records, images, by_image_id, val_ids, device)
        history.append({"epoch": epoch, "train_nll_per_batch_mean": epoch_loss / steps, "validation": val})
        if val["nll_per_question"] < best:
            best = val["nll_per_question"]
            torch.save({"model": model.state_dict(), "mode": args.mode, "epoch": epoch, "vocab_size": len(vocab), "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "retrieval_ckpt_sha256": hashlib.sha256(args.retrieval_ckpt.read_bytes()).hexdigest()}, out / "best.pt")
        print(json.dumps({"epoch": epoch, "train_nll": epoch_loss / steps, "val_nll": val["nll_per_question"], "val_acc": val["accuracy"], "best_nll": best}), flush=True)
    result = {
        "mode": args.mode,
        "source_retrieval_checkpoint": str(args.retrieval_ckpt),
        "source_retrieval_epoch": retrieval_ckpt["epoch"],
        "frozen_visual_weights": True,
        "train_yes_no": len(train_ids),
        "validation_yes_no": len(val_ids),
        "test_used": False,
        "train_yes_fraction": sum(records[i]["answer"] == "yes" for i in train_ids) / len(train_ids),
        "validation_yes_fraction": sum(records[i]["answer"] == "yes" for i in val_ids) / len(val_ids),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "elapsed_seconds": time.time() - start_time,
        "selection": "smallest internal validation binary NLL; test not loaded",
        "history": history,
    }
    (out / "train.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
