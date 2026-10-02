"""Train real binary Fashion image AE and VAE with an explicit Bernoulli target."""

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


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/c30_fashion_binary"
SEED = 20260924
LATENT = 16


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_and_images(device: torch.device):
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = []
    for name in ("train", "validation"):
        path = DATA / f"{name}.npy"
        if digest(path) != manifest["arrays"][name]["sha256"]:
            raise RuntimeError(f"binary Fashion source changed: {name}")
        data = np.load(path, allow_pickle=False)
        if (data.ndim != 3 or data.shape[1:] != (28, 28) or
                not np.isin(data, [0, 1]).all()):
            raise RuntimeError("binary Fashion shape/values changed")
        arrays.append(torch.from_numpy(data[:, None].copy()).to(
            device=device, dtype=torch.float32
        ))
    return arrays[0], arrays[1], {
        "manifest_sha256": digest(manifest_path),
        "train_npy_sha256": manifest["arrays"]["train"]["sha256"],
        "validation_npy_sha256": manifest["arrays"]["validation"]["sha256"],
        "threshold": manifest["threshold"],
    }


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        )

    def forward(self, images):
        return self.net(images)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(LATENT, 128), nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7), nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, latent):
        return self.upsample(
            self.linear(latent).reshape(-1, 64, 7, 7)
        )


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.code = nn.Linear(128, LATENT)
        self.decoder = Decoder()

    def forward(self, images):
        latent = self.code(self.encoder(images))
        return self.decoder(latent), latent


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mean = nn.Linear(128, LATENT)
        self.log_variance = nn.Linear(128, LATENT)
        self.decoder = Decoder()

    def forward(self, images, use_mean=False):
        hidden = self.encoder(images)
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden).clamp(-12, 10)
        latent = (
            mean if use_mean else
            mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
        )
        return self.decoder(latent), mean, log_variance


def losses(mode: str, model: nn.Module, images: torch.Tensor):
    if mode == "ae":
        logits, latent = model(images)
        kl = torch.zeros(len(images), device=images.device)
    else:
        logits, mean, log_variance = model(images)
        kl = 0.5 * (
            mean.square() + log_variance.exp() - 1 - log_variance
        ).sum(dim=1)
        latent = mean
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, images, reduction="none"
    ).flatten(1).sum(dim=1)
    total = reconstruction + kl
    return total.mean(), reconstruction.mean(), kl.mean(), latent


@torch.no_grad()
def validate(mode: str, model: nn.Module, images: torch.Tensor):
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(images.device) if images.device.type == "cuda" else None
    torch.manual_seed(SEED + 5000)
    if images.device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 5000)
    weighted = np.zeros(3, dtype=np.float64)
    latent_abs = 0.0
    for start in range(0, len(images), 256):
        batch = images[start:start + 256]
        total, reconstruction, kl, latent = losses(mode, model, batch)
        weighted += np.array([
            float(total), float(reconstruction), float(kl)
        ]) * len(batch)
        latent_abs += float(latent.abs().mean()) * len(batch)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, images.device)
    return {
        "loss_per_image": float(weighted[0] / len(images)),
        "reconstruction_bce_per_image": float(weighted[1] / len(images)),
        "reconstruction_bce_per_pixel": float(weighted[1] / len(images) / 784),
        "kl_per_image": float(weighted[2] / len(images)),
        "latent_mean_absolute_coordinate": latent_abs / len(images),
    }


def train(mode: str, epochs: int, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, val_images, source = source_and_images(device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    model = (AE() if mode == "ae" else VAE()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    order_rng = np.random.default_rng(SEED)
    initial = validate(mode, model, val_images)
    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(len(train_images))
        running = np.zeros(3, dtype=np.float64)
        seen = 0
        for start in range(0, len(order), 256):
            chosen = order[start:start + 256]
            total, reconstruction, kl, _ = losses(
                mode, model, train_images[chosen]
            )
            if not torch.isfinite(total):
                raise RuntimeError("nonfinite AE/VAE loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += np.array([
                float(total.detach()), float(reconstruction.detach()),
                float(kl.detach()),
            ]) * len(chosen)
            seen += len(chosen)
        val = validate(mode, model, val_images)
        history.append({
            "epoch": epoch,
            "train_online_loss_per_image": float(running[0] / seen),
            "train_online_reconstruction_bce_per_image": float(running[1] / seen),
            "train_online_kl_per_image": float(running[2] / seen),
            "validation": val,
        })
        if val["loss_per_image"] < best_loss:
            best_loss = val["loss_per_image"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(mode, "epoch", epoch, "val", round(val["loss_per_image"], 3),
              "recon", round(val["reconstruction_bce_per_image"], 3),
              "KL", round(val["kl_per_image"], 3), flush=True)
    if best_state is None:
        raise RuntimeError("no AE/VAE selected state")
    checkpoint = {
        "mode": mode, "source_identity": source,
        "selected_epoch": best_epoch,
        "model": {name: value.cpu().clone() for name, value in best_state.items()},
    }
    best_path = out_dir / "best.pt"
    torch.save(checkpoint, best_path)
    report = {
        "scope": "real Fashion-MNIST official train images thresholded >=128 to binary; labels not loaded; official test not used",
        "mode": mode, "latent_dimension": LATENT,
        "source_identity": source,
        "epochs_planned": epochs, "batch_size": 256,
        "objective": (
            "deterministic AE Bernoulli cross-entropy reconstruction of binary pixels; no prior distribution specified"
            if mode == "ae" else
            "VAE negative one-sample ELBO: Bernoulli pixel BCE plus KL diagonal Gaussian q(z|x) to N(0,I), per image"
        ),
        "optimizer": "Adam lr0.001 gradient_clip5",
        "initial_validation": initial,
        "history": history,
        "selected_epoch": best_epoch,
        "selection": "minimum full 10k validation objective with fixed VAE epsilon seed; no class labels or official test",
        "checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest(best_path),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "labels_loaded": False,
        "official_test_opened": False,
    }
    (out_dir / "train.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("ae", "vae"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir or Path(f"work/runs/c30_fashion_{args.mode}")
    train(args.mode, args.epochs, ROOT / out)


if __name__ == "__main__":
    main()
