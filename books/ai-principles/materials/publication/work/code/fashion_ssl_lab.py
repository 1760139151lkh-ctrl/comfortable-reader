"""Label-free Fashion-MNIST representation training, then frozen linear probes.

Pretrain modes load only the official image bytes, never their class-label files.
The existing C13 split was originally class-stratified; that historical split
choice is recorded, while no labels enter the C29 pretraining forward/loss.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from prepare_fashion_mnist import load_images, load_labels


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/fashion_mnist"
SOURCE = DATA / "manifest.json"
SPLIT = DATA / "chapter_split.npz"
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preserve existing result: {path}")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_identity() -> dict:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    image_files = ("train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz")
    for name in image_files:
        expected = manifest["files"][name]["sha256"]
        if digest(DATA / name) != expected:
            raise RuntimeError(f"official image source changed: {name}")
    return {
        "c13_manifest_sha256": digest(SOURCE),
        "c13_split_sha256": digest(SPLIT),
        "train_image_sha256": digest(DATA / image_files[0]),
        "official_test_image_sha256": digest(DATA / image_files[1]),
    }


def image_only_data(device: torch.device):
    identity = source_identity()
    all_images = load_images(DATA / "train-images-idx3-ubyte.gz", 60000)
    split = np.load(SPLIT)
    train = split["train_indices"].copy()
    validation = split["validation_indices"].copy()
    if len(train) != 50000 or len(validation) != 10000:
        raise RuntimeError("C13 split changed")
    pixels = torch.from_numpy(all_images[:, None].copy()).to(device=device, dtype=torch.float32) / 255
    return pixels, train, validation, identity


class Encoder(nn.Module):
    def __init__(self, channels: int = 1, side: int = 28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (side // 4) ** 2, 128),
                                  nn.ReLU())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(images))


class ContrastHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 64))

    def forward(self, features):
        return F.normalize(self.net(features), dim=-1)


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(128, 28 * 28)

    def forward(self, features):
        return self.net(features).reshape(-1, 1, 28, 28)


def two_views(images: torch.Tensor) -> torch.Tensor:
    count = len(images)
    side = images.shape[-1]
    pad = 2 if side == 28 else 4
    padded = F.pad(images, (pad, pad, pad, pad))
    dy = torch.randint(-pad, pad + 1, (count,), device=images.device)
    dx = torch.randint(-pad, pad + 1, (count,), device=images.device)
    rows = torch.arange(side, device=images.device)[None, :] + pad + dy[:, None]
    cols = torch.arange(side, device=images.device)[None, :] + pad + dx[:, None]
    moved = padded[
        torch.arange(count, device=images.device)[:, None, None, None],
        torch.arange(images.shape[1], device=images.device)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]
    flip = torch.rand(count, device=images.device) < 0.5
    moved[flip] = moved[flip].flip(-1)
    brightness = 0.85 + 0.30 * torch.rand(count, 1, 1, 1, device=images.device)
    return (moved * brightness + 0.025 * torch.randn_like(moved)).clamp(0, 1)


def mask_blocks(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(images)
    block = torch.rand(count, 1, 7, 7, device=images.device) < 0.5
    mask = block.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
    return images.masked_fill(mask, 0), mask


def contrast_loss(z: torch.Tensor, batch_size: int, temperature: float = 0.2):
    scores = z @ z.T / temperature
    scores.fill_diagonal_(float("-inf"))
    partner = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (2 * batch_size)
    return F.cross_entropy(scores, partner)


def objective(mode: str, encoder: Encoder, task_head: nn.Module,
              images: torch.Tensor) -> torch.Tensor:
    if mode in ("contrast", "contrast_wrong"):
        a = two_views(images)
        b = two_views(images)
        if mode == "contrast_wrong":
            # Same images and augmentation draws, but the nominated positive
            # is from a different source. The true partner remains a negative.
            b = b.roll(1, dims=0)
        z = task_head(encoder(torch.cat((a, b))))
        return contrast_loss(z, len(images))
    if mode == "masked":
        hidden, mask = mask_blocks(images)
        guess = task_head(encoder(hidden))
        return ((guess - images).square() * mask).sum() / mask.sum().clamp_min(1)
    raise ValueError(mode)


@torch.no_grad()
def validation_objective(mode: str, encoder, task_head, images, indices):
    encoder.eval()
    task_head.eval()
    # Fixed augmentation/mask randomness across epochs, restore train RNG after.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(images.device) if images.device.type == "cuda" else None
    torch.manual_seed(SEED + 5000)
    if images.device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 5000)
    totals = []
    for start in range(0, min(len(indices), 2048), 256):
        chosen = indices[start:start + 256]
        if len(chosen) < 2:
            continue
        totals.append(float(objective(mode, encoder, task_head, images[chosen])))
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, images.device)
    return float(np.mean(totals))


@torch.no_grad()
def feature_spread(encoder, images, indices):
    encoder.eval()
    chosen = indices[:2048]
    parts = [encoder(images[chosen[start:start + 256]]) for start in range(0, len(chosen), 256)]
    features = torch.cat(parts)
    std = features.float().std(dim=0)
    return {"mean_coordinate_std": float(std.mean()),
            "coordinates_below_0.01_std": int((std < 0.01).sum()),
            "feature_width": int(features.shape[1])}


def train(mode: str, epochs: int, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, train_rows, validation_rows, source = image_only_data(device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    encoder = Encoder().to(device)
    task_head = (ContrastHead() if mode.startswith("contrast") else MaskDecoder()).to(device)
    initial_encoder = copy.deepcopy(encoder.state_dict())
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(task_head.parameters()),
        lr=0.001, weight_decay=1e-4,
    )
    rng = np.random.default_rng(SEED)
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        encoder.train()
        task_head.train()
        order = rng.permutation(train_rows)
        running = 0.0
        steps = 0
        for start in range(0, len(order), 256):
            chosen = order[start:start + 256]
            if len(chosen) < 2:
                continue
            loss = objective(mode, encoder, task_head, images[chosen])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite SSL loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(task_head.parameters()), 1.0
            )
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        val = validation_objective(mode, encoder, task_head, images, validation_rows)
        spread = feature_spread(encoder, images, validation_rows)
        entry = {"epoch": epoch, "train_online_mean_loss": running / steps,
                 "validation_fixed_views_loss": val, "steps": steps,
                 "feature_spread": spread}
        history.append(entry)
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "task_head": copy.deepcopy(task_head.state_dict()),
            }
        print(mode, "epoch", epoch, "online", round(entry["train_online_mean_loss"], 4),
              "val", round(val, 4), "feature_std", round(spread["mean_coordinate_std"], 4),
              flush=True)
    if best_state is None:
        raise RuntimeError("no selected state")
    checkpoint = {
        "mode": mode, "selected_epoch": best_epoch,
        "encoder": {key: value.cpu().clone() for key, value in best_state["encoder"].items()},
        "task_head": {key: value.cpu().clone() for key, value in best_state["task_head"].items()},
        "initial_encoder": {key: value.cpu().clone() for key, value in initial_encoder.items()},
        "source_identity": source,
    }
    best_path = out_dir / "best.pt"
    torch.save(checkpoint, best_path)
    report = {
        "scope": "label-free real Fashion-MNIST image pretraining; C13 stratified split indices reused but no labels file loaded in this mode",
        "mode": mode, "epochs_planned": epochs, "batch_images": 256,
        "source_identity": source,
        "training_objective": (
            "two independently shifted/flipped/brightness-noised views per original; "
            "2B NT-Xent with positive same source and other sources negative; tau0.2"
            if mode == "contrast" else
            "identical view generation and contrastive loss, but second-view batch cyclically shifted by one: nominated positives come from different original images; intentionally incorrect relation"
            if mode == "contrast_wrong" else
            "half of 4x4 image patches masked; global-encoder/linear-decoder predicts original pixels; MSE only masked pixels, simplified masked AE not MAE"
        ),
        "optimizer": "AdamW lr0.001 weight_decay1e-4 gradient_clip1",
        "selection": "lowest fixed-augmentation/mask self-supervised loss on first2048 C13 validation rows; labels unseen",
        "history": history,
        "selected_epoch": best_epoch,
        "checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest(best_path),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "labels_loaded_in_pretraining": False,
        "official_test_opened": False,
    }
    write_json(out_dir / "train.json", report)
    print("selected", best_epoch, "val", best_val)


@torch.no_grad()
def extract(encoder, images: torch.Tensor, indices: np.ndarray) -> np.ndarray:
    encoder.eval()
    values = []
    for start in range(0, len(indices), 512):
        values.append(encoder(images[indices[start:start + 512]]).float().cpu().numpy())
    return np.concatenate(values)


def probe(contrast_dir: Path, masked_dir: Path, wrong_dir: Path | None, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"preserve existing probe: {out}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, train_rows, validation_rows, source = image_only_data(device)
    # Only training/validation labels enter until every linear probe is selected.
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    for label_file in ("train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"):
        if digest(DATA / label_file) != manifest["files"][label_file]["sha256"]:
            raise RuntimeError(f"official label source changed: {label_file}")
    train_labels = load_labels(DATA / "train-labels-idx1-ubyte.gz", 60000)
    subset = []
    rng = np.random.default_rng(SEED + 1)
    for label in range(10):
        available = train_rows[train_labels[train_rows] == label]
        subset.extend(rng.choice(available, 500, replace=False).tolist())
    subset = np.array(sorted(subset), dtype=np.int32)
    encoders = {}
    torch.manual_seed(SEED)
    encoders["random_same_architecture"] = Encoder().to(device)
    identities = {}
    sources = [
        ("contrast", contrast_dir, "contrast"),
        ("masked_reconstruction", masked_dir, "masked"),
    ]
    if wrong_dir is not None:
        sources.append(("contrast_wrong_pair", wrong_dir, "contrast_wrong"))
    for name, directory, expected_mode in sources:
        report = json.loads((directory / "train.json").read_text(encoding="utf-8"))
        path = ROOT / report["checkpoint"]
        if (report["mode"] != expected_mode or report["source_identity"] != source
                or digest(path) != report["checkpoint_sha256"]):
            raise RuntimeError(f"{name} source/checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["mode"] != expected_mode or state["source_identity"] != source:
            raise RuntimeError(f"{name} checkpoint metadata changed")
        model = Encoder().to(device)
        model.load_state_dict(state["encoder"])
        encoders[name] = model
        identities[name] = report["checkpoint_sha256"]
    results = {}
    selected_models = {}
    for name, encoder in encoders.items():
        raw_train = extract(encoder, images, subset)
        raw_val = extract(encoder, images, validation_rows)
        raw_spread = float(raw_val.std(axis=0).mean())
        scaler = StandardScaler().fit(raw_train)
        x_train = scaler.transform(raw_train)
        x_val = scaler.transform(raw_val)
        best_acc = -1
        selected = None
        candidates = []
        for c in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
            clf.fit(x_train, train_labels[subset])
            accuracy = float((clf.predict(x_val) == train_labels[validation_rows]).mean())
            candidates.append({"C": c, "validation_accuracy": accuracy,
                               "max_solver_iterations": int(clf.n_iter_.max())})
            if accuracy > best_acc:
                selected, best_acc, chosen_c = clf, accuracy, c
        if selected is None:
            raise RuntimeError("no linear probe")
        selected_models[name] = (encoder, selected, scaler)
        results[name] = {
            "labeled_train_count": len(subset),
            "validation_count": len(validation_rows),
            "C_candidates": candidates, "selected_C": chosen_c,
            "validation_accuracy": best_acc,
            "feature_coordinate_std_mean_validation_raw": raw_spread,
            "probe_feature_standardization": "mean/std fitted on frozen features of 500 labeled-train images per class, no validation/test fit",
        }
    # C13 used this official test earlier; it is reopened here only after choices.
    official_test = torch.from_numpy(
        load_images(DATA / "t10k-images-idx3-ubyte.gz", 10000)[:, None].copy()
    ).to(device=device, dtype=torch.float32) / 255
    test_labels = load_labels(DATA / "t10k-labels-idx1-ubyte.gz", 10000)
    for name, (encoder, selected, scaler) in selected_models.items():
        x_test = scaler.transform(extract(encoder, official_test, np.arange(10000)))
        test_accuracy = float((selected.predict(x_test) == test_labels).mean())
        results[name]["official_test_count"] = len(test_labels)
        results[name]["official_test_accuracy"] = test_accuracy
        print(name, "val", round(results[name]["validation_accuracy"], 4),
              "test", round(test_accuracy, 4),
              flush=True)
    write_json(out, {
        "scope": "frozen random/contrast/masked and optional wrong-pair Fashion encoders; only linear head sees 500 labels per class after SSL. Official Fashion test was already seen by C13 and is reused, not fresh author blind.",
        "source_identity": source, "ssl_checkpoints": identities,
        "official_label_sha256": {
            name: digest(DATA / name)
            for name in ("train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
        },
        "label_budget": "500 from each of 10 C13 training classes, selected after freezing; C13 validation chooses logistic C",
        "results": results, "labels_loaded_only_in_probe": True,
    })


@torch.no_grad()
def masked_baseline(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"preserve existing result: {out}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, train_rows, validation_rows, source = image_only_data(device)
    train_mean_image = images[train_rows].mean(dim=0, keepdim=True)
    train_single_mean = images[train_rows].mean()
    torch.manual_seed(SEED + 5000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 5000)
    totals = {"zero": [], "single_mean": [], "per_pixel_mean": []}
    for start in range(0, 2048, 256):
        original = images[validation_rows[start:start + 256]]
        _, mask = mask_blocks(original)
        denominator = mask.sum().clamp_min(1)
        for name, guess in (
            ("zero", torch.zeros_like(original)),
            ("single_mean", train_single_mean),
            ("per_pixel_mean", train_mean_image),
        ):
            totals[name].append(float(((guess - original).square() * mask).sum() / denominator))
    write_json(out, {
        "scope": "label-free fixed-mask pixel-MSE baselines on first2048 C13 validation images; same masking seed/blocks as SSL validation",
        "source_identity": source,
        "train_images_for_mean": len(train_rows),
        "validation_images": 2048,
        "mean_masked_mse": {key: float(np.mean(values)) for key, values in totals.items()},
        "labels_loaded": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "probe", "masked-baseline"))
    parser.add_argument("--objective", choices=("contrast", "contrast_wrong", "masked"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--contrast-dir", type=Path,
                        default=Path("work/runs/c29_fashion_contrast"))
    parser.add_argument("--masked-dir", type=Path,
                        default=Path("work/runs/c29_fashion_masked"))
    parser.add_argument("--wrong-dir", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("work/results/c29_fashion_linear_probe.json"))
    args = parser.parse_args()
    if args.mode == "train":
        if args.objective is None:
            parser.error("--objective is required for train")
        directory = args.out_dir or Path("work/runs/c29_fashion_" + args.objective)
        train(args.objective, args.epochs, ROOT / directory)
    elif args.mode == "probe":
        probe(ROOT / args.contrast_dir, ROOT / args.masked_dir,
              ROOT / args.wrong_dir if args.wrong_dir else None, ROOT / args.out)
    else:
        masked_baseline(ROOT / args.out)


if __name__ == "__main__":
    main()
