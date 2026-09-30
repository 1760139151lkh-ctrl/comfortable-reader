"""From-random visual/category model for a controlled real-photo yes/no task."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from c35_image_text_retrieval import DATA, ImageEncoder

OBJECT = Path("work/data/c35_object_presence")
CLASSES = ["person", "car", "bottle", "cup", "chair", "dining table"]


class ObjectQuestionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = ImageEncoder(128)
        self.category = nn.Embedding(len(CLASSES), 128)
        self.bias = nn.Parameter(torch.zeros(len(CLASSES)))

    def forward(self, x, class_id):
        visual = self.image(x)
        ask = self.category(class_id)
        return (visual * ask).sum(dim=-1) / 128**0.5 + self.bias[class_id]


def batch(records, indices, images, by_image, device):
    sample = [records[i] for i in indices]
    x = torch.from_numpy(np.asarray(images[[by_image[r["image_id"]] for r in sample]], dtype=np.float32).copy()).to(device) / 255.0
    category = torch.tensor([CLASSES.index(r["category"]) for r in sample], device=device)
    target = torch.tensor([float(r["label"]) for r in sample], device=device)
    return x, category, target


@torch.inference_mode()
def evaluate(model, records, indices, images, by_image, device):
    model.eval()
    by_category = {name: {"loss": 0.0, "correct": 0, "n": 0} for name in CLASSES}
    for start in range(0, len(indices), 128):
        ids = indices[start:start + 128]
        x, c, y = batch(records, ids, images, by_image, device)
        logit = model(x, c)
        losses = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        for local, idx in enumerate(ids):
            item = by_category[records[idx]["category"]]
            item["loss"] += float(losses[local])
            item["correct"] += int((logit[local] >= 0).float() == y[local])
            item["n"] += 1
    per_category = {name: {"questions": info["n"], "nll": info["loss"] / info["n"], "accuracy": info["correct"] / info["n"]} for name, info in by_category.items()}
    return {"macro_nll": sum(x["nll"] for x in per_category.values()) / len(CLASSES), "macro_accuracy": sum(x["accuracy"] for x in per_category.values()) / len(CLASSES), "total_accuracy": sum(x["accuracy"] * x["questions"] for x in per_category.values()) / len(indices), "per_category": per_category}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise RuntimeError("nonempty run dir")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((DATA / "image_records.json").read_text(encoding="utf-8"))
    by_image = {r["image_id"]: i for i, r in enumerate(rows)}
    images = np.load(DATA / f"images_{source['image_side']}_chw_uint8.npy", mmap_mode="r")
    records = json.loads((OBJECT / "records.json").read_text(encoding="utf-8"))
    train_ids = [i for i, r in enumerate(records) if r["split"] == "train"]
    val_ids = [i for i, r in enumerate(records) if r["split"] == "validation"]
    counts = Counter(records[i]["category"] for i in train_ids)
    weights = {name: len(train_ids) / (len(CLASSES) * n) for name, n in counts.items()}
    model = ObjectQuestionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    history, best = [], float("inf")
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        random.shuffle(train_ids)
        sum_loss, steps = 0.0, 0
        for start in range(0, len(train_ids), 128):
            ids = train_ids[start:start + 128]
            x, c, y = batch(records, ids, images, by_image, device)
            logits = model(x, c)
            loss_each = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            w = torch.tensor([weights[records[i]["category"]] for i in ids], device=device)
            loss = (loss_each * w).sum() / w.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.detach())
            steps += 1
        val = evaluate(model, records, val_ids, images, by_image, device)
        history.append({"epoch": epoch, "train_weighted_batch_nll_mean": sum_loss / steps, "validation": val})
        if val["macro_nll"] < best:
            best = val["macro_nll"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "data_manifest_sha256": hashlib.sha256((OBJECT / "manifest.json").read_bytes()).hexdigest()}, a.out_dir / "best.pt")
        print(json.dumps({"epoch": epoch, "train": sum_loss / steps, "val_macro_nll": val["macro_nll"], "val_macro_accuracy": val["macro_accuracy"], "best": best}), flush=True)
    report = {"identity": "COCO official instance label derived, author fixed-template six-category questions, image encoder randomly initialized (not external CLIP)", "train_questions": len(train_ids), "validation_questions": len(val_ids), "test_used": False, "selection": "smallest validation macro category NLL; test never used in train", "trainable_parameters": sum(p.numel() for p in model.parameters()), "elapsed_seconds": time.time() - t0, "history": history}
    (a.out_dir / "train.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
