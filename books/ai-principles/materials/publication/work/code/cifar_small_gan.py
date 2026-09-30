"""Train a small unconditional CIFAR-10 GAN from random weights, no class labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_ssl_lab import ROOT, train_pixels


SEED = 20260924
LATENT = 128


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        spectral = nn.utils.spectral_norm
        self.net = nn.Sequential(
            spectral(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral(nn.Conv2d(256, 1, 4, 1, 0)),
        )

    def forward(self, images):
        return self.net(images).flatten()


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)


@torch.no_grad()
def uint8_samples(generator, noise):
    generator.eval()
    output = ((generator(noise) + 1) * 127.5).clamp(0, 255)
    return output.permute(0, 2, 3, 1).byte().cpu().numpy()


def train(epochs: int, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing GAN run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixels, train_rows, val_rows, source = train_pixels(device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    generator, discriminator = Generator().to(device), Discriminator().to(device)
    generator.apply(initialize)
    # Spectral normalization has wrapped each discriminator convolution;
    # initialize its underlying original parameter, not the computed weight.
    for layer in discriminator.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight_orig, 0, 0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    fixed_random = torch.Generator(device="cpu").manual_seed(SEED + 9000)
    fixed_noise = torch.randn(
        64, LATENT, 1, 1, generator=fixed_random
    ).to(device)
    preview = out_dir / "preview_epoch_00.npy"
    np.save(preview, uint8_samples(generator, fixed_noise), allow_pickle=False)
    previews = [{"epoch": 0,
                 "path": str(preview.relative_to(ROOT)).replace("\\", "/"),
                 "sha256": digest(preview)}]
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        order = rng.permutation(train_rows)
        sum_d, sum_g, sum_real, sum_fake, seen = 0.0, 0.0, 0.0, 0.0, 0
        for start in range(0, len(order), 128):
            rows = order[start:start + 128]
            real = pixels[rows] * 2 - 1
            batch_size = len(rows)
            noise = torch.randn(batch_size, LATENT, 1, 1, device=device)
            fake = generator(noise)
            real_logit = discriminator(real)
            fake_logit = discriminator(fake.detach())
            d_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()
            # A new generated batch at the updated D; the non-saturating
            # generator objective increases D(G(noise)), not a log density.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            discriminator.eval()  # also freezes spectral-norm power iteration
            new_noise = torch.randn(batch_size, LATENT, 1, 1, device=device)
            generated = generator(new_noise)
            g_loss = F.softplus(-discriminator(generated)).mean()
            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            discriminator.train()
            if not torch.isfinite(d_loss) or not torch.isfinite(g_loss):
                raise RuntimeError("nonfinite GAN loss")
            sum_d += float(d_loss.detach()) * batch_size
            sum_g += float(g_loss.detach()) * batch_size
            sum_real += float(real_logit.detach().sigmoid().mean()) * batch_size
            sum_fake += float(fake_logit.detach().sigmoid().mean()) * batch_size
            seen += batch_size
        entry = {
            "epoch": epoch,
            "online_discriminator_loss_mean": sum_d / seen,
            "online_nonsaturating_generator_loss_mean": sum_g / seen,
            "online_discriminator_probability_real_mean": sum_real / seen,
            "online_discriminator_probability_fake_mean": sum_fake / seen,
        }
        history.append(entry)
        print("cifar GAN epoch", epoch, "D", round(entry["online_discriminator_loss_mean"], 4),
              "G", round(entry["online_nonsaturating_generator_loss_mean"], 4),
              flush=True)
        if epoch in {5, 10, 15, epochs}:
            sample_path = out_dir / f"preview_epoch_{epoch:02d}.npy"
            np.save(sample_path, uint8_samples(generator, fixed_noise),
                    allow_pickle=False)
            previews.append({
                "epoch": epoch,
                "path": str(sample_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest(sample_path),
            })
    final_path = out_dir / "final.pt"
    torch.save({
        "generator": {k: v.cpu().clone() for k, v in generator.state_dict().items()},
        "discriminator": {k: v.cpu().clone() for k, v in discriminator.state_dict().items()},
        "source_identity": source,
        "epochs": epochs,
    }, final_path)
    real_validation = (pixels[val_rows[:64]] * 255).round().clamp(0, 255)
    real_uint8 = real_validation.permute(0, 2, 3, 1).byte().cpu().numpy()
    generated_uint8 = np.load(ROOT / previews[-1]["path"], allow_pickle=False)
    real_path = out_dir / "real_validation_64.npy"
    np.save(real_path, real_uint8, allow_pickle=False)
    report = {
        "scope": "official CIFAR-10 32x32 RGB pixels, random unstratified C29 45k/5k split; no class labels or official test images opened",
        "source_identity": source,
        "architecture": "small transposed-convolution generator; spectral-normalized convolutional discriminator, no pretrained weights",
        "prior": "128 independent N(0,1) coordinates",
        "objective": "discriminator binary source log loss; non-saturating generator -log D(G(z)); no tractable p_G(x) computed",
        "optimizer": "Adam lr0.0002 beta1=0.5 beta2=0.999 for both networks",
        "epochs_planned": epochs,
        "batch_size": 128,
        "selection": "final predeclared epoch, not selected by D/G training loss or official test",
        "history": history,
        "fixed_noise_previews": previews,
        "final_checkpoint": str(final_path.relative_to(ROOT)).replace("\\", "/"),
        "final_checkpoint_sha256": digest(final_path),
        "real_validation_sample": str(real_path.relative_to(ROOT)).replace("\\", "/"),
        "real_validation_sample_sha256": digest(real_path),
        "real_validation_channel_mean": real_uint8.mean(axis=(0, 1, 2)).tolist(),
        "generated_channel_mean": generated_uint8.mean(axis=(0, 1, 2)).tolist(),
        "real_validation_channel_std": real_uint8.std(axis=(0, 1, 2)).tolist(),
        "generated_channel_std": generated_uint8.std(axis=(0, 1, 2)).tolist(),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "labels_loaded": False,
        "official_test_opened_for_this_experiment": False,
    }
    (out_dir / "train.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("work/runs/c30_cifar_small_gan"))
    args = parser.parse_args()
    train(args.epochs, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
