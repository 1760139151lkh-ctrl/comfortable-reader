"""A real binary Fashion RBM trained with explicit CD-1, with free Gibbs samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/c30_fashion_binary"
SEED = 20260924
VISIBLE = 784
HIDDEN = 128


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return x.sigmoid()


@torch.no_grad()
def reconstruct_bce(visible: torch.Tensor, W, b, c) -> float:
    total, count = 0.0, 0
    for start in range(0, len(visible), 256):
        actual = visible[start:start + 256]
        hidden_prob = sigmoid(actual @ W + c)
        pixel_prob = sigmoid(hidden_prob @ W.T + b).clamp(1e-6, 1 - 1e-6)
        total += float(F.binary_cross_entropy(
            pixel_prob, actual, reduction="sum"
        ))
        count += actual.numel()
    return total / count


def train(epochs: int, learning_rate: float, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"preserve existing RBM run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train", "validation"):
        if digest(DATA / f"{name}.npy") != manifest["arrays"][name]["sha256"]:
            raise RuntimeError(f"C30 binary source changed: {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = torch.from_numpy(
        np.load(DATA / "train.npy", allow_pickle=False).reshape(-1, VISIBLE).copy()
    ).to(device=device, dtype=torch.float32)
    validation = torch.from_numpy(
        np.load(DATA / "validation.npy", allow_pickle=False)[:2048].reshape(-1, VISIBLE).copy()
    ).to(device=device, dtype=torch.float32)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    W = 0.01 * torch.randn(VISIBLE, HIDDEN, device=device)
    frequency = train_data.mean(dim=0).clamp(0.01, 0.99)
    b = torch.logit(frequency)   # Data-derived visible bias, no class labels.
    c = torch.zeros(HIDDEN, device=device)
    rng = np.random.default_rng(SEED)
    initial_validation = reconstruct_bce(validation, W, b, c)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_data))
        online_recon_sum, online_count = 0.0, 0
        for start in range(0, len(order), 256):
            v0 = train_data[order[start:start + 256]]
            h0_prob = sigmoid(v0 @ W + c)
            h0 = torch.bernoulli(h0_prob)
            v1_prob = sigmoid(h0 @ W.T + b)
            v1 = torch.bernoulli(v1_prob)
            h1_prob = sigmoid(v1 @ W + c)
            # Positive phase is conditional expectation given clamped data;
            # negative phase is only ONE Gibbs transition, not equilibrium.
            scale = learning_rate / len(v0)
            delta_W = v0.T @ h0_prob - v1.T @ h1_prob
            delta_b = (v0 - v1).sum(dim=0)
            delta_c = (h0_prob - h1_prob).sum(dim=0)
            W.add_(scale * delta_W)
            b.add_(scale * delta_b)
            c.add_(scale * delta_c)
            online_recon_sum += float(F.binary_cross_entropy(
                v1_prob.clamp(1e-6, 1 - 1e-6), v0, reduction="sum"
            ))
            online_count += v0.numel()
        validation_recon = reconstruct_bce(validation, W, b, c)
        history.append({
            "epoch": epoch,
            "online_cd1_one_step_reconstruction_bce_per_pixel": (
                online_recon_sum / online_count
            ),
            "validation_mean_field_one_step_reconstruction_bce_per_pixel": validation_recon,
            "weight_frobenius_norm": float(torch.linalg.vector_norm(W)),
        })
        print("rbm epoch", epoch, "val recon", round(validation_recon, 4),
              flush=True)
    model_path = out_dir / "final.pt"
    torch.save({
        "W": W.cpu(), "b": b.cpu(), "c": c.cpu(),
        "source_manifest_sha256": digest(DATA / "manifest.json"),
        "epochs": epochs, "learning_rate": learning_rate,
    }, model_path)
    # Free-running chain begins from noise, not an observed training image.
    samples = torch.bernoulli(
        torch.full((64, VISIBLE), 0.5, device=device)
    )
    for _ in range(500):
        hidden = torch.bernoulli(sigmoid(samples @ W + c))
        samples = torch.bernoulli(sigmoid(hidden @ W.T + b))
    generated = samples.cpu().numpy().astype(np.uint8).reshape(64, 28, 28)
    sample_path = out_dir / "free_gibbs_500.npy"
    np.save(sample_path, generated, allow_pickle=False)
    report = {
        "scope": "official Fashion images thresholded binary; no labels or official test used; final predeclared epoch, no exact likelihood",
        "source_manifest_sha256": digest(DATA / "manifest.json"),
        "mode": "RBM CD-1, 784 visible/128 hidden binary units",
        "epochs_planned": epochs,
        "batch_size": 256,
        "learning_rate": learning_rate,
        "visible_bias_initialization": "logit of 50k training pixel frequency clipped to [0.01,0.99], label-free",
        "negative_phase": "one Gibbs step from each clamped training image, NOT equilibrium model sample",
        "initial_validation_reconstruction_bce_per_pixel": initial_validation,
        "history": history,
        "checkpoint": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest(model_path),
        "generated_sample_path": str(sample_path.relative_to(ROOT)).replace("\\", "/"),
        "generated_sample_sha256": digest(sample_path),
        "free_chain_start": "64 independent all-pixel Bernoulli(0.5) states",
        "free_chain_steps": 500,
        "free_samples_foreground_fraction": float(generated.mean()),
        "free_samples_pairwise_hamming_mean": float(
            np.mean([
                np.not_equal(generated[i], generated[j]).mean()
                for i in range(16) for j in range(i + 1, 16)
            ])
        ),
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
    }
    (out_dir / "train.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("work/runs/c30_fashion_rbm_cd1"))
    args = parser.parse_args()
    train(args.epochs, args.learning_rate, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
