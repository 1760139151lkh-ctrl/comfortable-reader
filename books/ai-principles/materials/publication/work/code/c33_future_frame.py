"""Predict a genuine Penn frame five indices ahead from two earlier frames."""

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


def batch(triplets: np.ndarray, indices: np.ndarray, device: torch.device,
          augment: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.asarray(triplets[indices], dtype=np.float32) / 255.0
    data = torch.from_numpy(data.transpose(0, 1, 4, 2, 3).copy()).to(device)
    if augment:
        flip = torch.rand(len(indices), device=device) < 0.5
        data[flip] = data[flip].flip(-1)
    return data[:, 0], data[:, 1], data[:, 2]


class FutureNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(6, 32, 5, padding=2), nn.GELU())
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
                                   nn.Conv2d(128, 128, 3, padding=1), nn.GELU())
        self.up1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(32, 3, 3, padding=1))

    def forward(self, past: torch.Tensor,
                current: torch.Tensor) -> torch.Tensor:
        start = self.in_conv(torch.cat([past, current], dim=1))
        half = self.down1(start)
        quarter = self.down2(half)
        up = F.interpolate(quarter, size=half.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up1(torch.cat([up, half], dim=1))
        up = F.interpolate(up, size=start.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.up2(torch.cat([up, start], dim=1))
        return torch.clamp(current + 0.5 * torch.tanh(delta), 0.0, 1.0)


@torch.no_grad()
def evaluate(model: FutureNet, triplets: np.ndarray, device: torch.device,
             batch_size: int) -> dict:
    model.eval()
    count_pixels = 0
    model_sse = 0.0
    copy_sse = 0.0
    moving_model_sse = 0.0
    moving_copy_sse = 0.0
    moving_pixels = 0
    for start in range(0, len(triplets), batch_size):
        indexes = np.arange(start, min(start + batch_size, len(triplets)))
        past, current, future = batch(triplets, indexes, device)
        prediction = model(past, current)
        model_error = (prediction - future).square()
        copy_error = (current - future).square()
        model_sse += model_error.sum().item()
        copy_sse += copy_error.sum().item()
        count_pixels += future.numel()
        # This future-dependent mask is an EVALUATION diagnostic only. Neither
        # the network nor copy baseline receives it as an input or train label.
        moving = (current - future).abs().mean(dim=1, keepdim=True) > 0.08
        moving_model_sse += (model_error * moving).sum().item()
        moving_copy_sse += (copy_error * moving).sum().item()
        moving_pixels += int(moving.sum()) * future.shape[1]
    return {
        "model_mse_per_rgb_coordinate": model_sse / count_pixels,
        "copy_current_mse_per_rgb_coordinate": copy_sse / count_pixels,
        "moving_region_fraction_of_pixels": moving_pixels / count_pixels,
        "moving_region_model_mse_per_coordinate": moving_model_sse / moving_pixels if moving_pixels else None,
        "moving_region_copy_mse_per_coordinate": moving_copy_sse / moving_pixels if moving_pixels else None,
        "triplets": len(triplets),
        "moving_mask_definition": "mean RGB absolute difference between actual future and current >0.08; diagnostic uses truth after prediction, not a model input",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing next-frame run: {out}")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["official_test_labels_or_frames_loaded"]:
        raise RuntimeError("this training data opened official test")
    for part in ("train", "validation"):
        if sha256(DATA / f"{part}_triplets.npy") != manifest["triplet_npy_sha256"][part]:
            raise RuntimeError(f"real-frame cache changed: {part}")
    train = np.load(DATA / "train_triplets.npy", mmap_mode="r", allow_pickle=False)
    validation = np.load(DATA / "validation_triplets.npy", mmap_mode="r", allow_pickle=False)
    out.mkdir(parents=True)
    random.seed(20260924)
    np.random.seed(20260924)
    torch.manual_seed(20260924)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260924)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FutureNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    initial = evaluate(model, validation, device, args.batch_size)
    best_mse, best_epoch = float("inf"), None
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = np.random.permutation(len(train))
        running, images = 0.0, 0
        for start in range(0, len(permutation), args.batch_size):
            indexes = permutation[start:start + args.batch_size]
            past, current, future = batch(train, indexes, device, augment=True)
            prediction = model(past, current)
            loss = F.mse_loss(prediction, future)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(indexes)
            images += len(indexes)
        result = evaluate(model, validation, device, args.batch_size)
        history.append({"epoch": epoch, "train_mse_per_rgb_coordinate": running / images,
                        "validation": result})
        if result["model_mse_per_rgb_coordinate"] < best_mse:
            best_mse = result["model_mse_per_rgb_coordinate"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "data_manifest_sha256": sha256(manifest_path)},
                       out / "best.pt")
        if epoch % 5 == 0 or epoch == 1:
            print("future epoch", epoch, "train", round(running / images, 5),
                  "val", round(result["model_mse_per_rgb_coordinate"], 5),
                  "copy", round(result["copy_current_mse_per_rgb_coordinate"], 5),
                  flush=True)
    report = {
        "task": "deterministic frame at t+5 from real Penn RGB frames t-5,t, selected on official TRAIN videos with visible monotonic wrist movement",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "data_manifest_sha256": sha256(manifest_path),
        "code_sha256": sha256(Path(__file__)),
        "official_test_accessed": False,
        "train_triplets": len(train), "validation_triplets": len(validation),
        "frame_gap_indices": 5,
        "architecture": "small 6-channel residual encoder-decoder, no action/text label input and no stochastic latent",
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": "per RGB coordinate mean squared error of predicted t+5 frame",
        "optimizer": "AdamW lr0.001 weight_decay0.01",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "device": str(device), "torch_version": torch.__version__,
        "initial_validation": initial,
        "best_epoch_by_validation_model_mse": best_epoch,
        "best_checkpoint_sha256": sha256(out / "best.pt"),
        "history": history,
        "wall_seconds": time.perf_counter() - start_time,
        "scope_limit": "one predicted frame on author-filtered real clips, not an unrolled video, not a calibrated physical simulator; low MSE may reward static background or blurry averages",
    }
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
