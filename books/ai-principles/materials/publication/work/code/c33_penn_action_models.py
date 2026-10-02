"""Compare real-video action classifiers with one, unordered, or ordered frames."""

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
DATA = ROOT / "work/data/penn_action_original/action_8frames_v1"
MODES = ("single", "mean", "temporal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrameEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(),
            nn.Linear(96 * 2 * 3, 128), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionModel(nn.Module):
    def __init__(self, mode: str, classes: int = 15) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.encoder = FrameEncoder()
        if mode == "temporal":
            self.temporal = nn.Sequential(
                nn.Conv1d(128, 128, 3, padding=1), nn.GELU(),
                nn.Conv1d(128, 128, 3, padding=1), nn.GELU(),
                nn.Flatten(), nn.Linear(128 * 8, 128), nn.GELU(),
            )
        else:
            self.temporal = nn.Identity()
        self.head = nn.Linear(128, classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: batch x 8 frames x 3 channels x 90 height x 160 width.
        batch, frames, channels, height, width = video.shape
        if frames != 8 or channels != 3:
            raise ValueError("expected eight RGB frames")
        if self.mode == "single":
            features = self.encoder(video[:, 4])
        else:
            features = self.encoder(video.reshape(batch * frames, channels, height, width))
            features = features.reshape(batch, frames, 128)
            if self.mode == "mean":
                features = features.mean(dim=1)
            else:
                features = self.temporal(features.transpose(1, 2))
        return self.head(features)


def batch_tensor(frames: np.ndarray, indices: np.ndarray, device: torch.device,
                 augment: bool = False) -> torch.Tensor:
    array = np.asarray(frames[indices], dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(0, 1, 4, 2, 3).copy()).to(device)
    if augment:
        flip = torch.rand(len(indices), device=device) < 0.5
        tensor[flip] = tensor[flip].flip(-1)
    return tensor


@torch.no_grad()
def evaluate(model: ActionModel, frames: np.ndarray, rows: list[dict],
             names: list[str], batch_size: int, device: torch.device) -> dict:
    model.eval()
    lookup = {name: i for i, name in enumerate(names)}
    total_loss, correct = 0.0, 0
    predictions = []
    for start in range(0, len(rows), batch_size):
        indexes = np.arange(start, min(start + batch_size, len(rows)))
        x = batch_tensor(frames, indexes, device)
        y = torch.tensor([lookup[rows[i]["action"]] for i in indexes],
                         dtype=torch.long, device=device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        guess = logits.argmax(dim=1)
        correct += int((guess == y).sum())
        predictions.extend(guess.cpu().tolist())
    return {"mean_cross_entropy": total_loss / len(rows),
            "accuracy": correct / len(rows), "correct": correct,
            "count": len(rows), "predictions": predictions}


def fit_one(mode: str, frames: np.ndarray, rows: list[dict], names: list[str],
            out: Path, epochs: int, batch_size: int, device: torch.device,
            manifest_sha: str) -> dict:
    torch.manual_seed(20260924)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260924)
    np.random.seed(20260924)
    train_ids = np.array([i for i, row in enumerate(rows)
                          if row["author_partition"] == "train"], dtype=int)
    val_ids = np.array([i for i, row in enumerate(rows)
                        if row["author_partition"] == "validation"], dtype=int)
    if len(set(train_ids) & set(val_ids)) or not len(train_ids) or not len(val_ids):
        raise RuntimeError("bad internal split")
    val_frames = np.asarray(frames[val_ids])
    val_rows = [rows[i] for i in val_ids]
    model = ActionModel(mode, len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    history = []
    best_loss, best_epoch = float("inf"), None
    initial = evaluate(model, val_frames, val_rows, names, batch_size, device)
    start_time = time.perf_counter()
    lookup = {name: i for i, name in enumerate(names)}
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = np.random.permutation(train_ids)
        total_loss, seen = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            x = batch_tensor(frames, indices, device, augment=True)
            y = torch.tensor([lookup[rows[i]["action"]] for i in indices],
                             dtype=torch.long, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
            seen += len(indices)
        validation = evaluate(model, val_frames, val_rows, names, batch_size, device)
        history.append({"epoch": epoch,
                        "train_mean_cross_entropy": total_loss / seen,
                        "validation_mean_cross_entropy": validation["mean_cross_entropy"],
                        "validation_accuracy": validation["accuracy"]})
        if validation["mean_cross_entropy"] < best_loss:
            best_loss = validation["mean_cross_entropy"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "mode": mode,
                        "class_names": names, "epoch": epoch,
                        "data_manifest_sha256": manifest_sha},
                       out / (mode + "_best.pt"))
        if epoch % 5 == 0 or epoch == 1:
            print(mode, "epoch", epoch, "train", round(total_loss / seen, 4),
                  "val", round(validation["mean_cross_entropy"], 4),
                  "accuracy", round(validation["accuracy"], 4), flush=True)
    return {"mode": mode,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_clips": len(train_ids), "validation_clips": len(val_ids),
            "initial_validation": {k: v for k, v in initial.items() if k != "predictions"},
            "best_epoch_by_validation_cross_entropy": best_epoch,
            "best_validation_cross_entropy": best_loss,
            "best_checkpoint_sha256": sha256(out / (mode + "_best.pt")),
            "history": history,
            "wall_seconds": time.perf_counter() - start_time}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing Penn action run: {out}")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["official_test_frames_or_action_labels_loaded"]:
        raise RuntimeError("official test used before model selection")
    frames_path = DATA / "train_source_frames.npy"
    if sha256(frames_path) != manifest["cached_array_sha256"]:
        raise RuntimeError("Penn action frame cache changed")
    frames = np.load(frames_path, mmap_mode="r", allow_pickle=False)
    rows = manifest["rows"]
    if len(frames) != len(rows) or len(rows) != 1258:
        raise RuntimeError("Penn action training rows changed")
    out.mkdir(parents=True)
    random.seed(20260924)
    np.random.seed(20260924)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for mode in MODES:
        results.append(fit_one(mode, frames, rows, manifest["class_names"],
                               out, args.epochs, args.batch_size, device,
                               sha256(manifest_path)))
    report = {
        "task": "Penn original 15 human-action classes from genuine eight-frame RGB clips; one-frame vs orderless average vs ordered temporal convolution",
        "official_benchmark_replication": False,
        "source_archive_sha256": manifest["source_archive_sha256"],
        "data_manifest_sha256": sha256(manifest_path),
        "code_sha256": sha256(Path(__file__)),
        "official_test_accessed": False,
        "frame_selection": manifest["frame_sampling"],
        "model_modes": MODES,
        "common_frame_encoder": "three stride-2 2-D conv blocks then 128-dimensional embedding, randomly initialized for every mode",
        "single": "only fifth of eight sampled frames seen by model",
        "mean": "all eight per-frame embeddings averaged without order",
        "temporal": "all eight ordered per-frame embeddings through two 1-D temporal convs then position-aware head",
        "train_augment": "joint horizontal flip of all frames, action class label unchanged",
        "optimizer": "AdamW lr0.001 weight_decay0.01",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "device": str(device), "torch_version": torch.__version__,
        "results": results,
        "comparison_limit": "mode parameter counts and multiply-add costs differ; accuracy differences are local, not proof of universal temporal architecture superiority",
    }
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
