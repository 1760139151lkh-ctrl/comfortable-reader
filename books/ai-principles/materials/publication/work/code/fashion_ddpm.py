"""Small class-conditional Gaussian DDPM trained from official Fashion gray pixels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_fashion_mnist import load_images, load_labels


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/fashion_mnist"
SEED = 20260924
TIMESTEPS = 300
NULL_LABEL = 10
BETAS = torch.linspace(0.0001, 0.05, TIMESTEPS, dtype=torch.float64)
ALPHAS = 1 - BETAS
BAR = torch.cat((torch.ones(1, dtype=torch.float64), torch.cumprod(ALPHAS, dim=0)))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_data(device):
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
        if digest(DATA / name) != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"official Fashion file changed: {name}")
    original = load_images(DATA / "train-images-idx3-ubyte.gz", 60000)
    labels = load_labels(DATA / "train-labels-idx1-ubyte.gz", 60000)
    split_path = DATA / "chapter_split.npz"
    split = np.load(split_path)
    train_rows = split["train_indices"].copy()
    validation_rows = split["validation_indices"].copy()
    if len(train_rows) != 50000 or len(validation_rows) != 10000:
        raise RuntimeError("Fashion C13 split changed")
    images = torch.from_numpy(original[:, None].copy()).to(
        device=device, dtype=torch.float32
    ) / 127.5 - 1
    class_labels = torch.from_numpy(labels.copy()).to(
        device=device, dtype=torch.long
    )
    identity = {
        "official_manifest_sha256": digest(manifest_path),
        "train_images_sha256": digest(DATA / "train-images-idx3-ubyte.gz"),
        "train_labels_sha256": digest(DATA / "train-labels-idx1-ubyte.gz"),
        "c13_stratified_split_sha256": digest(split_path),
        "train_count": len(train_rows), "validation_count": len(validation_rows),
        "pixel_transform": "official grayscale uint8 /127.5 -1, no binarization",
        "labels": "10 official Fashion classes enter conditional model; null condition randomly trained",
    }
    return images, class_labels, train_rows, validation_rows, identity


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.emb = nn.Linear(embedding_width, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels else
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, images, embedding):
        hidden = F.silu(self.norm1(self.conv1(images)))
        hidden = hidden + self.emb(embedding)[:, :, None, None]
        hidden = self.norm2(self.conv2(F.silu(hidden)))
        return F.silu(hidden + self.skip(images))


class NoiseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        width = 128
        self.time_embedding = nn.Embedding(TIMESTEPS + 1, width)
        self.class_embedding = nn.Embedding(NULL_LABEL + 1, width)
        self.embedding = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.input = nn.Conv2d(1, 32, 3, padding=1)
        self.top = ResBlock(32, 32, width)
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.middle1 = ResBlock(64, 64, width)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.middle2 = ResBlock(128, 128, width)
        self.bottleneck = ResBlock(128, 128, width)
        self.up1 = ResBlock(128 + 64, 64, width)
        self.up2 = ResBlock(64 + 32, 32, width)
        self.output_norm = nn.GroupNorm(8, 32)
        self.output = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, noisy, timesteps, labels):
        embedding = self.embedding(
            self.time_embedding(timesteps) + self.class_embedding(labels)
        )
        high = self.top(self.input(noisy), embedding)
        middle = self.middle1(self.down1(high), embedding)
        low = self.middle2(self.down2(middle), embedding)
        low = self.bottleneck(low, embedding)
        middle_back = F.interpolate(
            low, size=middle.shape[-2:], mode="nearest"
        )
        middle_back = self.up1(
            torch.cat((middle_back, middle), dim=1), embedding
        )
        high_back = F.interpolate(
            middle_back, size=high.shape[-2:], mode="nearest"
        )
        high_back = self.up2(
            torch.cat((high_back, high), dim=1), embedding
        )
        return self.output(F.silu(self.output_norm(high_back)))


def time_coefficients(device):
    abar = BAR.to(device=device, dtype=torch.float32)
    beta = BETAS.to(device=device, dtype=torch.float32)
    alpha = ALPHAS.to(device=device, dtype=torch.float32)
    return abar, beta, alpha


def noise_batch(clean, timesteps, epsilon, abar):
    signal = abar[timesteps].sqrt()[:, None, None, None]
    noise = (1 - abar[timesteps]).sqrt()[:, None, None, None]
    return signal * clean + noise * epsilon


@torch.no_grad()
def validate(model, images, labels, rows, abar, count=2048):
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(images.device) if images.device.type == "cuda" else None
    torch.manual_seed(SEED + 5000)
    if images.device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 5000)
    total, by_group = 0.0, [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    for start in range(0, min(len(rows), count), 128):
        index = rows[start:start + 128]
        clean = images[index]
        step = torch.randint(1, TIMESTEPS + 1, (len(index),), device=images.device)
        epsilon = torch.randn_like(clean)
        noisy = noise_batch(clean, step, epsilon, abar)
        guessed = model(noisy, step, labels[index])
        per_image = (guessed - epsilon).square().flatten(1).mean(dim=1)
        total += float(per_image.sum())
        for group, chosen in enumerate(
            (step <= 100, (step > 100) & (step <= 200), step > 200)
        ):
            if chosen.any():
                by_group[group] += float(per_image[chosen].sum())
                counts[group] += int(chosen.sum())
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, images.device)
    actual = min(len(rows), count)
    return {
        "conditional_noise_mse": total / actual,
        "early_middle_late_mse": [
            by_group[i] / counts[i] if counts[i] else None for i in range(3)
        ],
        "early_middle_late_counts": counts,
        "validation_images": actual,
    }


@torch.no_grad()
def sample(model, device, label: int, guidance: float, count: int, seed: int,
           abar, beta, alpha):
    model.eval()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    current = torch.randn(count, 1, 28, 28, device=device)
    class_label = torch.full((count,), label, device=device, dtype=torch.long)
    null_label = torch.full((count,), NULL_LABEL, device=device, dtype=torch.long)
    for t in range(TIMESTEPS, 0, -1):
        step = torch.full((count,), t, device=device, dtype=torch.long)
        condition = model(current, step, class_label)
        if guidance != 0 and label != NULL_LABEL:
            unconditional = model(current, step, null_label)
            epsilon = (1 + guidance) * condition - guidance * unconditional
        else:
            epsilon = condition
        mean = (
            current - beta[t - 1] / torch.sqrt(1 - abar[t]) * epsilon
        ) / torch.sqrt(alpha[t - 1])
        if t > 1:
            posterior_variance = (
                beta[t - 1] * (1 - abar[t - 1]) / (1 - abar[t])
            )
            current = mean + torch.sqrt(
                posterior_variance.clamp_min(0)
            ) * torch.randn_like(current)
        else:
            current = mean
    raw_min, raw_max = float(current.min()), float(current.max())
    pixels = ((current.clamp(-1, 1) + 1) * 127.5).round().byte()
    return pixels[:, 0].cpu().numpy(), raw_min, raw_max


def train(epochs: int, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing DDPM run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, train_rows, val_rows, source = source_data(device)
    abar, beta, alpha = time_coefficients(device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    model = NoiseUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003,
                                   weight_decay=1e-4)
    initial = validate(model, images, labels, val_rows, abar)
    order_rng = np.random.default_rng(SEED)
    history = []
    best_val, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(train_rows)
        total, seen, null_count = 0.0, 0, 0
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            clean = images[index]
            step = torch.randint(1, TIMESTEPS + 1, (len(index),), device=device)
            epsilon = torch.randn_like(clean)
            noisy = noise_batch(clean, step, epsilon, abar)
            actual_labels = labels[index].clone()
            null = torch.rand(len(index), device=device) < 0.1
            actual_labels[null] = NULL_LABEL
            null_count += int(null.sum())
            guess = model(noisy, step, actual_labels)
            loss = F.mse_loss(guess, epsilon)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite diffusion noise MSE")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        validation = validate(model, images, labels, val_rows, abar)
        history.append({
            "epoch": epoch,
            "train_online_noise_mse": total / seen,
            "train_null_condition_count": null_count,
            "validation": validation,
        })
        if validation["conditional_noise_mse"] < best_val:
            best_val = validation["conditional_noise_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print("fashion ddpm epoch", epoch,
              "val", round(validation["conditional_noise_mse"], 5),
              "early/mid/late",
              [round(v, 4) for v in validation["early_middle_late_mse"]],
              flush=True)
    if best_state is None:
        raise RuntimeError("no diffusion checkpoint chosen")
    model.load_state_dict(best_state)
    checkpoint = out_dir / "best.pt"
    torch.save({
        "model": {name: value.cpu().clone() for name, value in best_state.items()},
        "selected_epoch": best_epoch,
        "source_identity": source,
        "timesteps": TIMESTEPS,
        "beta_first_last": [float(BETAS[0]), float(BETAS[-1])],
    }, checkpoint)
    samples = []
    for label, guidance, name in (
        (NULL_LABEL, 0.0, "unconditional"),
        (9, 0.0, "ankle_boot_w0"),
        (9, 2.0, "ankle_boot_w2"),
        (9, 5.0, "ankle_boot_w5"),
    ):
        generated, raw_min, raw_max = sample(
            model, device, label, guidance, count=16, seed=SEED + 9000,
            abar=abar, beta=beta, alpha=alpha
        )
        path = out_dir / f"samples_{name}.npy"
        np.save(path, generated, allow_pickle=False)
        samples.append({
            "name": name, "label": label, "guidance_w": guidance,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(path),
            "raw_before_display_clip_min_max": [raw_min, raw_max],
        })
    report = {
        "scope": "official Fashion 28x28 original grayscale conditional denoising, C13 stratified old train/validation; 10 official classes intentionally loaded, null condition trained; no official test",
        "source_identity": source,
        "mode": "small 300-step class-conditional DDPM epsilon predictor",
        "timesteps": TIMESTEPS,
        "beta_schedule": "linear 0.0001 to 0.05",
        "bar_alpha_terminal": float(BAR[-1]),
        "q_terminal_conditioned_on_x0": "sqrt(bar_alpha_T)*x0+sqrt(1-bar_alpha_T)*normal; only approximately pure normal",
        "train_target": "epsilon MSE at uniformly sampled step and real image, mean over batch/pixels; not exact unweighted negative log likelihood",
        "null_condition_probability": 0.1,
        "batch_size": 128,
        "epochs_planned": epochs,
        "optimizer": "AdamW lr0.0003 weight_decay1e-4 gradient_clip1",
        "initial_validation": initial,
        "history": history,
        "selection": "minimum conditional fixed-noise/timestep MSE on first2048 C13 validation rows, not image quality",
        "selected_epoch": best_epoch,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest(checkpoint),
        "reverse_sampler": "300 ancestral Gaussian steps, posterior variance beta_t*(1-bar_alpha_(t-1))/(1-bar_alpha_t), final step deterministic; no image clipping inside chain",
        "guidance_definition": "paper convention epsilon_w=(1+w)*epsilon_cond-w*epsilon_null; w0 ordinary class-conditional",
        "samples": samples,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "official_test_opened": False,
    }
    (out_dir / "train.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("work/runs/c31_fashion_ddpm"))
    args = parser.parse_args()
    train(args.epochs, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
