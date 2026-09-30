"""Small natural-caption conditional image VAE trained on real COCO pixels."""
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

from c35_image_text_retrieval import DATA, batch_tokens


class CaptionCondition(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 96, padding_idx=0)
        self.gru = nn.GRU(96, 128, batch_first=True)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]


class ConditionalVAE(nn.Module):
    def __init__(self, vocab_size: int, mode: str):
        super().__init__()
        self.mode = mode
        self.text = CaptionCondition(vocab_size)
        self.image = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 192, 4, 2, 1), nn.ReLU(), nn.Flatten(),
        )
        self.posterior = nn.Linear(192 * 4 * 4 + 128, 128)
        self.start = nn.Linear(64 + 128, 192 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def condition(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        text = self.text(tokens, lengths)
        if self.mode == "unconditional":
            text = torch.zeros_like(text)
        return text

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.start(torch.cat([z, condition], dim=-1)).reshape(-1, 192, 4, 4)
        return self.decoder(x)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor, noise: torch.Tensor | None = None):
        condition = self.condition(tokens, lengths)
        stats = self.posterior(torch.cat([self.image(x), condition], dim=-1))
        mean, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8, 8)
        if noise is None:
            noise = torch.randn_like(mean)
        z = mean + torch.exp(0.5 * logvar) * noise
        prediction = self.decode(z, condition)
        # Fixed Gaussian observation variance 0.2^2. Additive normalizer omitted
        # equally for both arms; RGB integer dequantization is NOT modeled.
        squared_per_image = (prediction - x).square().flatten(1).sum(dim=1)
        reconstruct = squared_per_image / (2 * 0.2**2)
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=-1)
        return prediction, reconstruct.mean(), kl.mean()


def image_batch(images: np.ndarray, indices: list[int], device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(images[indices], dtype=np.float32).copy()).to(device) / 255.0
    return F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)


@torch.inference_mode()
def evaluate(model, images, rows, ids, device):
    model.eval()
    rec, kl, count = 0.0, 0.0, 0
    generator = torch.Generator(device=device).manual_seed(35_000)
    for start in range(0, len(ids), 64):
        batch = ids[start:start + 64]
        x = image_batch(images, batch, device)
        tok, lengths = batch_tokens([rows[i]["caption_tokens"][0] for i in batch], device)
        noise = torch.randn((len(batch), 64), device=device, generator=generator)
        _, rec_loss, kl_loss = model(x, tok, lengths, noise)
        rec += float(rec_loss) * len(batch)
        kl += float(kl_loss) * len(batch)
        count += len(batch)
    return {"images": count, "reconstruction_fixed_variance_gaussian_negative_log_term_per_image": rec / count, "posterior_kl_per_image": kl / count, "negative_elbo_up_to_common_gaussian_normalizer": (rec + kl) / count, "mean_squared_error_per_rgb_channel": rec / count * (2 * 0.2**2) / (64 * 64 * 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["conditional", "unconditional"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise RuntimeError("nonempty run dir")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = json.loads((DATA / "image_records.json").read_text(encoding="utf-8"))
    images = np.load(DATA / f"images_{manifest['image_side']}_chw_uint8.npy", mmap_mode="r")
    vocab_size = len(json.loads((DATA / "vocab.json").read_text(encoding="utf-8")))
    train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val_ids = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    model = ConditionalVAE(vocab_size, a.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.0001)
    history, best = [], float("inf")
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        random.shuffle(train_ids)
        epoch_rec, epoch_kl, steps = 0.0, 0.0, 0
        for start in range(0, len(train_ids), 64):
            ids = train_ids[start:start + 64]
            x = image_batch(images, ids, device)
            captions = [rows[i]["caption_tokens"][(epoch + i) % len(rows[i]["caption_tokens"])] for i in ids]
            tok, lengths = batch_tokens(captions, device)
            _, rec, kl = model(x, tok, lengths)
            loss = rec + kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_rec += float(rec.detach())
            epoch_kl += float(kl.detach())
            steps += 1
        val = evaluate(model, images, rows, val_ids, device)
        history.append({"epoch": epoch, "train_rec_per_batch_mean": epoch_rec / steps, "train_kl_per_batch_mean": epoch_kl / steps, "validation": val})
        score = val["negative_elbo_up_to_common_gaussian_normalizer"]
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "mode": a.mode, "epoch": epoch, "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "vocab_size": vocab_size}, a.out_dir / "best.pt")
        print(json.dumps({"epoch": epoch, "train_rec": epoch_rec / steps, "train_kl": epoch_kl / steps, "val_objective": score, "best": best}), flush=True)
    report = {"mode": a.mode, "train_images": len(train_ids), "validation_images": len(val_ids), "test_used": False, "pixels": "COCO true RGB compressed from 112px aspect-padded cache to 64px, no labels", "sigma": 0.2, "latent_dimensions": 64, "parameters": sum(p.numel() for p in model.parameters()), "device": str(device), "elapsed_seconds": time.time() - t0, "selection": "minimum author-internal validation negative ELBO term; fixed validation posterior noise; test unopened", "history": history}
    (a.out_dir / "train.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
