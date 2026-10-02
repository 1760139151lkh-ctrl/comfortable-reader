"""Fit two non-speech ESC-10 classifiers without reading held-out fold 5."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/esc10_33c8ce9/chapter_features"
FEATURES = DATA / "logmel80"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows() -> tuple[dict, list[dict], list[dict]]:
    path = DATA / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    train = [row for row in manifest["rows"] if row["split"] == "train"]
    validation = [row for row in manifest["rows"] if row["split"] == "validation"]
    if len(train) != 240 or len(validation) != 80:
        raise RuntimeError("ESC split counts changed")
    return manifest, train, validation


def load_features(rows: list[dict]) -> np.ndarray:
    arrays = [np.load(FEATURES / (row["id"] + ".npy"),
                      allow_pickle=False).astype(np.float32) for row in rows]
    if len({array.shape for array in arrays}) != 1:
        raise RuntimeError("ESC-10 clips should have the same frame count")
    return np.stack(arrays)


def labels(rows: list[dict], names: list[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(names)}
    return np.array([lookup[row["category"]] for row in rows], dtype=np.int64)


class EventCNN(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(96, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, None])


def fit_baseline(out: Path) -> None:
    manifest, train, validation = read_rows()
    x_train = load_features(train)
    x_val = load_features(validation)
    train_summary = np.concatenate([x_train.mean(axis=1), x_train.std(axis=1)], axis=1)
    val_summary = np.concatenate([x_val.mean(axis=1), x_val.std(axis=1)], axis=1)
    names = manifest["class_names"]
    y_train, y_val = labels(train, names), labels(validation, names)
    scaler = StandardScaler().fit(train_summary)
    train_scaled = scaler.transform(train_summary)
    val_scaled = scaler.transform(val_summary)
    records = []
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=1000, random_state=20260924)
        model.fit(train_scaled, y_train)
        probs = model.predict_proba(val_scaled)
        nll = float(-np.log(np.maximum(probs[np.arange(len(y_val)), y_val], 1e-12)).mean())
        accuracy = float((probs.argmax(axis=1) == y_val).mean())
        records.append({"C": c, "validation_mean_negative_log_probability": nll,
                        "validation_accuracy": accuracy})
        if best is None or nll < best[0]:
            best = (nll, c, model)
    assert best is not None
    _, selected_c, selected_model = best
    np.savez(out / "baseline_weights.npz", scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_, coefficients=selected_model.coef_,
             intercept=selected_model.intercept_, classes=selected_model.classes_)
    report = {
        "task": "ESC-10 non-speech sound event classification, author one-fold split",
        "upstream_archive_sha256": manifest["upstream_archive_sha256"],
        "feature_manifest_sha256": sha256(DATA / "manifest.json"),
        "train_count": len(train), "validation_count": len(validation),
        "test_count_read": 0,
        "feature": "per-clip temporal mean and std of each of 80 log-mel bands",
        "method": "train-only StandardScaler + multinomial logistic regression",
        "class_names": names,
        "candidate_C": records,
        "selected_C_by_validation_nll": selected_c,
        "selected_validation_nll": best[0],
        "weights_sha256": sha256(out / "baseline_weights.npz"),
    }
    (out / "baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print("baseline selected C", selected_c, "val nll", round(best[0], 4), flush=True)


@torch.no_grad()
def evaluate_cnn(model: EventCNN, x: torch.Tensor, y: torch.Tensor,
                 batch_size: int) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size])
        target = y[start:start + batch_size]
        total_loss += F.cross_entropy(logits, target, reduction="sum").item()
        correct += int((logits.argmax(dim=1) == target).sum().item())
    return {"mean_cross_entropy": total_loss / len(x),
            "accuracy": correct / len(x), "correct": correct,
            "count": len(x)}


def fit_cnn(out: Path, epochs: int, batch_size: int) -> None:
    manifest, train, validation = read_rows()
    names = manifest["class_names"]
    x_train = load_features(train)
    x_val = load_features(validation)
    # Preserve relative frequency profile. The per-band mean/variance below
    # are estimated only from the 240 training clips, never from validation.
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.from_numpy((x_train - mean) / std).to(device)
    val_tensor = torch.from_numpy((x_val - mean) / std).to(device)
    y_train = torch.from_numpy(labels(train, names)).to(device)
    y_val = torch.from_numpy(labels(validation, names)).to(device)
    model = EventCNN(len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    initial = evaluate_cnn(model, val_tensor, y_val, batch_size)
    best = float("inf")
    best_epoch = None
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train), device=device)
        running = 0.0
        for start in range(0, len(train), batch_size):
            indexes = permutation[start:start + batch_size]
            logits = model(train_tensor[indexes])
            loss = F.cross_entropy(logits, y_train[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(indexes)
        result = evaluate_cnn(model, val_tensor, y_val, batch_size)
        history.append({"epoch": epoch, "train_mean_cross_entropy": running / len(train),
                        "validation": result})
        if result["mean_cross_entropy"] < best:
            best = result["mean_cross_entropy"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "mean": mean,
                        "std": std, "class_names": names, "epoch": epoch,
                        "feature_manifest_sha256": sha256(DATA / "manifest.json")},
                       out / "cnn_best.pt")
        if epoch % 5 == 0 or epoch == 1:
            print("event epoch", epoch, "train", round(running / len(train), 4),
                  "val", round(result["mean_cross_entropy"], 4),
                  "acc", round(result["accuracy"], 4), flush=True)
    report = {
        "task": "ESC-10 non-speech sound event classification, author one-fold split",
        "upstream_archive_sha256": manifest["upstream_archive_sha256"],
        "feature_manifest_sha256": sha256(DATA / "manifest.json"),
        "code_sha256": sha256(Path(__file__)),
        "train_count": len(train), "validation_count": len(validation),
        "test_count_read": 0,
        "seed": 20260924, "device": str(device), "torch_version": torch.__version__,
        "model": "three 3x3 strided 2D conv layers (24/48/96 channels), batch norm, global average, 10-class head",
        "parameters": sum(p.numel() for p in model.parameters()),
        "class_names": names,
        "feature_standardization": "per log-mel band train-only mean/std over train clips and frames; minimum std 0.2",
        "batch_size": batch_size, "epochs": epochs,
        "optimizer": "AdamW lr 0.001 weight_decay 0.01",
        "initial_validation": initial,
        "best_epoch_by_validation_cross_entropy": best_epoch,
        "best_checkpoint_sha256": sha256(out / "cnn_best.pt"),
        "history": history,
        "wall_seconds": time.perf_counter() - start_time,
    }
    (out / "cnn_train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing event run: {out}")
    out.mkdir(parents=True)
    random.seed(20260924)
    np.random.seed(20260924)
    torch.manual_seed(20260924)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260924)
    torch.set_num_threads(4)
    fit_baseline(out)
    fit_cnn(out, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
