"""Binary Fashion autoregressive image model with causal masked convolutions."""

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
SIDE = 28
PIXELS = SIDE * SIDE


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_kind):
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=kernel_size // 2)
        if mask_kind not in ("A", "B"):
            raise ValueError(mask_kind)
        middle = kernel_size // 2
        mask = torch.ones_like(self.weight)
        mask[:, :, middle + 1:, :] = 0
        mask[:, :, middle, middle + (0 if mask_kind == "A" else 1):] = 0
        self.register_buffer("causal_mask", mask)

    def forward(self, images):
        return F.conv2d(
            images, self.weight * self.causal_mask, self.bias,
            stride=self.stride, padding=self.padding,
        )


class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MaskedConv2d(1, 64, 7, "A"), nn.ReLU(),
            MaskedConv2d(64, 64, 3, "B"), nn.ReLU(),
            MaskedConv2d(64, 64, 3, "B"), nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, images):
        return self.net(images)


def data_on_device(device):
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = {}
    for name in ("train", "validation"):
        path = DATA / f"{name}.npy"
        if digest(path) != manifest["arrays"][name]["sha256"]:
            raise RuntimeError(f"C30 binary source changed: {name}")
        raw = np.load(path, allow_pickle=False)
        images[name] = torch.from_numpy(raw[:, None].copy()).to(
            device=device, dtype=torch.float32
        )
    source = {
        "manifest_sha256": digest(manifest_path),
        "train_npy_sha256": manifest["arrays"]["train"]["sha256"],
        "validation_npy_sha256": manifest["arrays"]["validation"]["sha256"],
    }
    return images["train"], images["validation"], source


@torch.no_grad()
def score(model, images):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(images), 256):
        batch = images[start:start + 256]
        logits = model(batch)
        total += float(F.binary_cross_entropy_with_logits(
            logits, batch, reduction="sum"
        ))
        count += batch.numel()
    return total / count


@torch.no_grad()
def causal_probe(model, device):
    model.eval()
    torch.manual_seed(SEED + 1234)
    original = torch.bernoulli(
        torch.full((1, 1, SIDE, SIDE), 0.5, device=device)
    )
    checks = []
    for place in (0, 79, 350, 783):
        altered = original.clone()
        altered.flatten()[place:] = 1 - altered.flatten()[place:]
        before = model(original).flatten()[:place + 1]
        after = model(altered).flatten()[:place + 1]
        difference = float((before - after).abs().max())
        checks.append({"first_changed_flat_pixel": place,
                       "earlier_or_same_logit_max_difference": difference})
        if difference > 1e-7:
            raise RuntimeError(f"future leakage at pixel {place}: {difference}")
    return checks


@torch.no_grad()
def generate(model, device, count=24):
    model.eval()
    torch.manual_seed(SEED + 8000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 8000)
    images = torch.zeros(count, 1, SIDE, SIDE, device=device)
    for place in range(PIXELS):
        probability = model(images).sigmoid().flatten(1)[:, place]
        sample = torch.bernoulli(probability)
        images.flatten(1)[:, place] = sample
    return images[:, 0].cpu().numpy().astype(np.uint8)


def train(epochs: int, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing PixelCNN run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training, validation, source = data_on_device(device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    model = PixelCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    causality_before = causal_probe(model, device)
    initial = score(model, validation)
    pixel_mean = training.mean(dim=0).clamp(1e-5, 1 - 1e-5)
    baseline = float(F.binary_cross_entropy(
        pixel_mean.expand_as(validation), validation,
        reduction="mean",
    ))
    order_rng = np.random.default_rng(SEED)
    history = []
    best_val, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(len(training))
        running, count = 0.0, 0
        for start in range(0, len(order), 256):
            batch = training[order[start:start + 256]]
            loss = F.binary_cross_entropy_with_logits(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite PixelCNN loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * batch.numel()
            count += batch.numel()
        validation_nll = score(model, validation)
        history.append({
            "epoch": epoch,
            "train_online_nll_per_pixel": running / count,
            "validation_nll_per_pixel": validation_nll,
        })
        if validation_nll < best_val:
            best_val, best_epoch = validation_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
        print("pixelcnn epoch", epoch, "val nll", round(validation_nll, 5),
              flush=True)
    if best_state is None:
        raise RuntimeError("no PixelCNN selected state")
    model.load_state_dict(best_state)
    causality_after = causal_probe(model, device)
    checkpoint_path = out_dir / "best.pt"
    torch.save({
        "model": {k: v.cpu().clone() for k, v in best_state.items()},
        "source_identity": source, "selected_epoch": best_epoch,
    }, checkpoint_path)
    generated = generate(model, device)
    sample_path = out_dir / "generated_24.npy"
    np.save(sample_path, generated, allow_pickle=False)
    report = {
        "scope": "binary official Fashion images, no class labels or official test; normalized joint on thresholded data, limited receptive field",
        "source_identity": source,
        "architecture": "7x7 mask A then two 3x3 mask B convolutions, width64; 1x1 Bernoulli logit",
        "limitation": "each pixel's conditional sees only a bounded raster-past receptive field, not all previous pixels",
        "epochs_planned": epochs,
        "batch_size": 256,
        "optimizer": "Adam lr0.001 gradient_clip5",
        "initial_validation_nll_per_pixel": initial,
        "train_pixelwise_unconditional_baseline_val_nll": baseline,
        "causal_probe_before": causality_before,
        "history": history,
        "selected_epoch": best_epoch,
        "selected_validation_nll_per_pixel": best_val,
        "causal_probe_after": causality_after,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest(checkpoint_path),
        "samples": str(sample_path.relative_to(ROOT)).replace("\\", "/"),
        "samples_sha256": digest(sample_path),
        "generation": "24 images, each 784 sequential Bernoulli draws; per-pixel logits recomputed from previous sampled pixels",
        "generated_foreground_fraction": float(generated.mean()),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
    }
    (out_dir / "train.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("work/runs/c30_fashion_pixelcnn"))
    args = parser.parse_args()
    train(args.epochs, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
