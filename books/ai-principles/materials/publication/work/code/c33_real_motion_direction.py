"""Learn image-plane wrist-direction signs from two genuine Penn frames.

The first/second example from each real triplet has the exact same CURRENT
frame and opposite direction labels. An isolated-current-frame classifier is
mathematically capped at 50% on paired evaluation. The image model below gets
two ordered RGB frames and their pixel difference, never the annotated joints.
"""

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


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/penn_action_original/motion_gap5_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_partition(name: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    path = DATA / f"{name}_triplets.npy"
    if sha256(path) != manifest["triplet_npy_sha256"][name]:
        raise RuntimeError(f"Penn triplet cache changed: {name}")
    triplets = np.load(path, mmap_mode="r", allow_pickle=False)
    metadata = manifest["metadata"][name]
    if len(triplets) != len(metadata) or triplets.shape[1:] != (3, 90, 160, 3):
        raise RuntimeError(f"Penn triplet cache shape changed: {name}")
    signs = np.array([row["past_to_current_label"] for row in metadata], dtype=np.int64)
    return triplets, signs, metadata


class DirectionCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(),
            nn.Linear(96 * 3 * 5, 96), nn.GELU(), nn.Linear(96, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def make_batch(triplets: np.ndarray, signs: np.ndarray, pair_ids: np.ndarray,
               device: torch.device, augment_flip: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor]:
    triplet_ids = pair_ids // 2
    side = pair_ids % 2
    first_indexes = np.where(side == 0, 0, 2)
    first = np.asarray(triplets[triplet_ids, first_indexes], dtype=np.float32) / 255.0
    current = np.asarray(triplets[triplet_ids, 1], dtype=np.float32) / 255.0
    delta = current - first
    input_array = np.concatenate([first, current, delta], axis=-1)
    x = torch.from_numpy(input_array.transpose(0, 3, 1, 2).copy()).to(device)
    y = torch.from_numpy((signs[triplet_ids] ^ side).astype(np.int64)).to(device)
    if augment_flip:
        flip = torch.rand(len(pair_ids), device=device) < 0.5
        x[flip] = x[flip].flip(-1)
        y[flip] = 1 - y[flip]
    return x, y


@torch.no_grad()
def evaluate(model: DirectionCNN, triplets: np.ndarray, signs: np.ndarray,
             batch_size: int, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    predictions = []
    for start in range(0, 2 * len(triplets), batch_size):
        pair_ids = np.arange(start, min(start + batch_size, 2 * len(triplets)))
        x, y = make_batch(triplets, signs, pair_ids, device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        predicted = logits.argmax(dim=1)
        correct += int((predicted == y).sum())
        predictions.extend(predicted.cpu().tolist())
    # For the paired single-current-frame baseline, each identical current
    # image occurs once with each opposite label, so accuracy is exactly 1/2.
    return {"mean_cross_entropy": total_loss / (2 * len(triplets)),
            "accuracy": correct / (2 * len(triplets)),
            "correct": correct, "count": 2 * len(triplets),
            "single_current_frame_optimal_accuracy_by_pair_symmetry": 0.5,
            "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing motion training: {out}")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["official_test_labels_or_frames_loaded"]:
        raise RuntimeError("training manifest already opened official test")
    train, signs_train, meta_train = load_partition("train")
    validation, signs_val, meta_val = load_partition("validation")
    if ({m["video_id"] for m in meta_train} & {m["video_id"] for m in meta_val}):
        raise RuntimeError("same video id in train and validation")
    out.mkdir(parents=True)
    random.seed(20260924)
    np.random.seed(20260924)
    torch.manual_seed(20260924)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260924)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=0.01)
    initial = evaluate(model, validation, signs_val, args.batch_size, device)
    best = float("inf")
    best_epoch = None
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = np.random.permutation(2 * len(train))
        total_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), args.batch_size):
            indexes = permutation[start:start + args.batch_size]
            x, y = make_batch(train, signs_train, indexes, device,
                              augment_flip=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indexes)
            seen += len(indexes)
        result = evaluate(model, validation, signs_val, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "validation_loss": result["mean_cross_entropy"],
                        "validation_accuracy": result["accuracy"]})
        if result["mean_cross_entropy"] < best:
            best = result["mean_cross_entropy"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "data_manifest_sha256": sha256(manifest_path)},
                       out / "best.pt")
        if epoch % 5 == 0 or epoch == 1:
            print("motion epoch", epoch, "train", round(total_loss / seen, 4),
                  "val", round(result["mean_cross_entropy"], 4),
                  "accuracy", round(result["accuracy"], 4), flush=True)
    report = {
        "task": "two-frame image-plane right-wrist horizontal direction from genuine Penn frames; author-derived balanced labels, not Penn's original action categories",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "data_manifest_sha256": sha256(manifest_path),
        "code_sha256": sha256(Path(__file__)),
        "official_test_accessed": False,
        "train_videos": len({m["video_id"] for m in meta_train}),
        "validation_videos": len({m["video_id"] for m in meta_val}),
        "train_pair_examples": 2 * len(train),
        "validation_pair_examples": 2 * len(validation),
        "baseline": "current-frame-only classifier gets exactly one of two opposite labels on each identical current image: 50%",
        "model": "9-channel ordered (first,current,current-first) image input, 3-layer small CNN with spatial head",
        "parameters": sum(p.numel() for p in model.parameters()),
        "horizontal_flip": "same flip applied to both frames and direction label inverted at train time only",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device), "torch_version": torch.__version__,
        "initial_validation": {k: v for k, v in initial.items() if k != "predictions"},
        "best_epoch_by_validation_loss": best_epoch,
        "best_checkpoint_sha256": sha256(out / "best.pt"),
        "history": history,
        "wall_seconds": time.perf_counter() - start_time,
    }
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
