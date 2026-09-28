"""Train a continuous-time straight-pair velocity field on real Fashion or CIFAR images."""

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

from cifar_ssl_lab import train_pixels as cifar_train_pixels
from fashion_ddpm import ResBlock
from prepare_fashion_mnist import load_images


ROOT = Path(__file__).resolve().parents[2]
FASHION = ROOT / "work/data/fashion_mnist"
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_data(dataset: str, device):
    if dataset == "cifar":
        images, train_rows, val_rows, source = cifar_train_pixels(device)
        source = {
            **source,
            "dataset": "official CIFAR-10 RGB, unstratified 45k/5k C29 split",
            "labels_loaded": False,
            "pixel_transform": "official uint8/127.5 -1",
        }
        return images * 2 - 1, train_rows, val_rows, source, 3, 32
    manifest_path = FASHION / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_path = FASHION / "train-images-idx3-ubyte.gz"
    if digest(image_path) != manifest["files"][image_path.name]["sha256"]:
        raise RuntimeError("Fashion official image source changed")
    split_path = FASHION / "chapter_split.npz"
    split = np.load(split_path)
    train_rows = split["train_indices"].copy()
    val_rows = split["validation_indices"].copy()
    if len(train_rows) != 50000 or len(val_rows) != 10000:
        raise RuntimeError("Fashion C13 split changed")
    raw = load_images(image_path, 60000)
    images = torch.from_numpy(raw[:, None].copy()).to(
        device=device, dtype=torch.float32
    ) / 127.5 - 1
    source = {
        "dataset": "official Fashion grayscale with C13 old class-stratified 50k/10k indices; labels not read by this program",
        "official_manifest_sha256": digest(manifest_path),
        "train_images_sha256": digest(image_path),
        "c13_split_sha256": digest(split_path),
        "labels_loaded": False,
        "pixel_transform": "official uint8/127.5 -1",
    }
    return images, train_rows, val_rows, source, 1, 28


class VelocityUNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        frequencies = 2 * math.pi * torch.tensor(
            [0.25, 0.5, 1, 2, 4, 8, 16, 32], dtype=torch.float32
        )
        self.register_buffer("time_frequencies", frequencies)
        self.time_embedding = nn.Sequential(
            nn.Linear(16, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.input = nn.Conv2d(channels, 32, 3, padding=1)
        self.top = ResBlock(32, 32, 128)
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.middle1 = ResBlock(64, 64, 128)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.middle2 = ResBlock(128, 128, 128)
        self.bottleneck = ResBlock(128, 128, 128)
        self.up1 = ResBlock(128 + 64, 64, 128)
        self.up2 = ResBlock(64 + 32, 32, 128)
        self.out_norm = nn.GroupNorm(8, 32)
        self.output = nn.Conv2d(32, channels, 3, padding=1)

    def forward(self, state, times):
        phase = times[:, None] * self.time_frequencies[None, :]
        embedding = self.time_embedding(
            torch.cat((phase.sin(), phase.cos()), dim=1)
        )
        high = self.top(self.input(state), embedding)
        middle = self.middle1(self.down1(high), embedding)
        low = self.middle2(self.down2(middle), embedding)
        low = self.bottleneck(low, embedding)
        middle_back = F.interpolate(low, size=middle.shape[-2:], mode="nearest")
        middle_back = self.up1(
            torch.cat((middle_back, middle), dim=1), embedding
        )
        high_back = F.interpolate(
            middle_back, size=high.shape[-2:], mode="nearest"
        )
        high_back = self.up2(
            torch.cat((high_back, high), dim=1), embedding
        )
        return self.output(F.silu(self.out_norm(high_back)))


@torch.no_grad()
def validate(model, images, val_rows, count=2048):
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(images.device) if images.device.type == "cuda" else None
    torch.manual_seed(SEED + 5000)
    if images.device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 5000)
    total, seen = 0.0, 0
    bins = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    for start in range(0, min(len(val_rows), count), 128):
        rows = val_rows[start:start + 128]
        target = images[rows]
        noise = torch.randn_like(target)
        time = torch.rand(len(rows), device=images.device)
        point = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * target
        speed = target - noise
        prediction = model(point, time)
        per_image = (prediction - speed).square().flatten(1).mean(dim=1)
        total += float(per_image.sum())
        seen += len(rows)
        for bin_id, selected in enumerate(
            (time < 1/3, (time >= 1/3) & (time < 2/3), time >= 2/3)
        ):
            if selected.any():
                bins[bin_id] += float(per_image[selected].sum())
                counts[bin_id] += int(selected.sum())
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, images.device)
    return {
        "velocity_mse_per_coordinate": total / seen,
        "early_middle_late_mse": [
            bins[i] / counts[i] if counts[i] else None for i in range(3)
        ],
        "early_middle_late_counts": counts,
        "validation_images": seen,
    }


@torch.no_grad()
def integrate(model, initial, steps: int, method: str):
    current = initial.clone()
    dt = 1 / steps
    for index in range(steps):
        time = torch.full(
            (len(current),), index * dt,
            device=current.device, dtype=torch.float32,
        )
        velocity = model(current, time)
        if method == "euler":
            current = current + dt * velocity
        elif method == "heun":
            proposed = current + dt * velocity
            next_time = torch.full_like(time, (index + 1) * dt)
            current = current + 0.5 * dt * (
                velocity + model(proposed, next_time)
            )
        else:
            raise ValueError(method)
    return current


def image_uint8(output, channels):
    pixels = ((output.clamp(-1, 1) + 1) * 127.5).round().byte()
    if channels == 1:
        return pixels[:, 0].cpu().numpy()
    return pixels.permute(0, 2, 3, 1).cpu().numpy()


def train(dataset: str, epochs: int, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing flow run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, train_rows, val_rows, source, channels, side = source_data(
        dataset, device
    )
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    model = VelocityUNet(channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0003, weight_decay=1e-4
    )
    initial_validation = validate(model, images, val_rows)
    order_rng = np.random.default_rng(SEED)
    history = []
    best_epoch, best_val, best_state = 0, float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(train_rows)
        online_sum, seen = 0.0, 0
        for start in range(0, len(order), 128):
            rows = order[start:start + 128]
            target = images[rows]
            noise = torch.randn_like(target)
            time = torch.rand(len(rows), device=device)
            point = ((1 - time[:, None, None, None]) * noise +
                     time[:, None, None, None] * target)
            velocity_target = target - noise
            velocity = model(point, time)
            loss = F.mse_loss(velocity, velocity_target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite flow MSE")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            online_sum += float(loss.detach()) * len(rows)
            seen += len(rows)
        result = validate(model, images, val_rows)
        history.append({
            "epoch": epoch,
            "train_online_velocity_mse": online_sum / seen,
            "validation": result,
        })
        if result["velocity_mse_per_coordinate"] < best_val:
            best_epoch, best_val = epoch, result["velocity_mse_per_coordinate"]
            best_state = copy.deepcopy(model.state_dict())
        print(dataset, "flow epoch", epoch, "val", round(best_val, 5),
              "current", round(result["velocity_mse_per_coordinate"], 5),
              flush=True)
    if best_state is None:
        raise RuntimeError("no flow checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = out_dir / "best.pt"
    torch.save({
        "model": {name: value.cpu().clone() for name, value in best_state.items()},
        "selected_epoch": best_epoch,
        "dataset": dataset, "source_identity": source,
    }, checkpoint)
    torch.manual_seed(SEED + 9000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 9000)
    fixed_noise = torch.randn(16, channels, side, side, device=device)
    generated = {}
    raw_output = {}
    sample_records = []
    for method, steps in (
        ("euler", 4), ("euler", 16), ("euler", 64),
        ("heun", 16), ("heun", 64)
    ):
        output = integrate(model, fixed_noise, steps, method)
        name = f"{method}_{steps}"
        raw_output[name] = output
        path = out_dir / f"samples_{name}.npy"
        np.save(path, image_uint8(output, channels), allow_pickle=False)
        generated[name] = path
        sample_records.append({
            "name": name, "method": method, "steps": steps,
            "network_evaluations": (
                steps if method == "euler" else 2 * steps
            ),
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(path),
            "raw_before_display_clip_min_max": [
                float(output.min()), float(output.max())
            ],
        })
    base = raw_output["heun_64"]
    differences = {
        name: float((candidate - base).square().mean().sqrt())
        for name, candidate in raw_output.items() if name != "heun_64"
    }
    real = image_uint8(images[val_rows[:16]], channels)
    real_path = out_dir / "real_validation_16.npy"
    np.save(real_path, real, allow_pickle=False)
    report = {
        "scope": f"official {dataset} real pixels and documented existing split, no class labels or official test; linear independent noise/data pair and approximate numerical ODE",
        "source_identity": source,
        "dataset": dataset,
        "channels": channels, "side": side,
        "target": "independent z~N(0,I), real x1, t~U[0,1], x_t=(1-t)z+t*x1, velocity target=x1-z",
        "model": "continuous Fourier time-conditioned small U-Net",
        "epochs_planned": epochs, "batch_size": 128,
        "optimizer": "AdamW lr0.0003 weight_decay1e-4 gradient_clip1",
        "initial_validation": initial_validation,
        "history": history,
        "selection": "minimum fixed-pair/time velocity MSE on first2048 validation images; no image quality or official test used",
        "selected_epoch": best_epoch,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest(checkpoint),
        "samplers": sample_records,
        "fixed_noise_seed": SEED + 9000,
        "raw_output_rmse_vs_heun64": differences,
        "real_validation_16_path": str(real_path.relative_to(ROOT)).replace("\\", "/"),
        "real_validation_16_sha256": digest(real_path),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "official_test_opened": False,
        "boundary": "independent straight pairs are training targets, learned vector field only sees current state/time; not the paper's full OT/positive-endpoint-variance or reflow setup",
    }
    (out_dir / "train.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fashion", "cifar"),
                        default="fashion")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir or Path(f"work/runs/c31_{args.dataset}_flow")
    train(args.dataset, args.epochs, ROOT / out)


if __name__ == "__main__":
    main()
