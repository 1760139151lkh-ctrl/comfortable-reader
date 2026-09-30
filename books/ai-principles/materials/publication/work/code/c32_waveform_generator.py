"""A small unconditional autoregressive μ-law speech-waveform model.

It predicts the next 8-kHz quantized sample from preceding samples. It has no
text encoder, duration model or external audio weights; it is not TTS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import soundfile as sf

from prepare_c32_wave_generation import decode_mulaw, encode_mulaw


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/librispeech_dev_clean/wavegen_speaker1272"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SampleGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, 64)
        self.gru = nn.GRU(64, 160, num_layers=2, batch_first=True,
                          dropout=0.1)
        self.head = nn.Linear(160, 256)

    def forward(self, codes: torch.Tensor,
                state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        y, state = self.gru(self.embedding(codes), state)
        return self.head(y), state


@torch.no_grad()
def evaluate(model: SampleGRU, data: torch.Tensor,
             batch_size: int) -> float:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for start in range(0, len(data), batch_size):
        chunk = data[start:start + batch_size]
        logits, _ = model(chunk[:, :-1])
        total_loss += F.cross_entropy(logits.reshape(-1, 256),
                                      chunk[:, 1:].reshape(-1),
                                      reduction="sum").item()
        total_targets += chunk[:, 1:].numel()
    return total_loss / total_targets


def train(out: Path, epochs: int, batch_size: int) -> None:
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (sha256(DATA / "train.npy") != manifest["train_npy_sha256"] or
            sha256(DATA / "validation.npy") != manifest["validation_npy_sha256"]):
        raise RuntimeError("wave generation data changed")
    train_codes = np.load(DATA / "train.npy", allow_pickle=False)
    val_codes = np.load(DATA / "validation.npy", allow_pickle=False)
    if train_codes.shape != (5373, 513) or val_codes.shape != (2091, 513):
        raise RuntimeError("wave generation sample counts changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(train_codes.astype(np.int64)).to(device)
    x_val = torch.from_numpy(val_codes.astype(np.int64)).to(device)
    counts = np.bincount(train_codes[:, 1:].reshape(-1), minlength=256).astype(np.float64)
    probs = (counts + 1) / (counts.sum() + 256)
    unigram_nll = float(-np.log(probs[val_codes[:, 1:].reshape(-1)]).mean())
    model = SampleGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    initial = evaluate(model, x_val, batch_size)
    best = initial
    best_epoch = 0
    torch.save({"model": model.state_dict(), "epoch": 0,
                "source_manifest_sha256": sha256(manifest_path)}, out / "best.pt")
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        total_targets = 0
        epoch_start = time.perf_counter()
        for start in range(0, len(x_train), batch_size):
            chunk = x_train[permutation[start:start + batch_size]]
            logits, _ = model(chunk[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, 256),
                                   chunk[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * chunk[:, 1:].numel()
            total_targets += chunk[:, 1:].numel()
        val_nll = evaluate(model, x_val, batch_size)
        record = {"epoch": epoch, "train_nll_per_sample": total_loss / total_targets,
                  "validation_nll_per_sample": val_nll,
                  "elapsed_seconds": time.perf_counter() - epoch_start}
        history.append(record)
        if val_nll < best:
            best = val_nll
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "source_manifest_sha256": sha256(manifest_path)}, out / "best.pt")
        print("wave epoch", epoch, "train", round(record["train_nll_per_sample"], 4),
              "val", round(val_nll, 4), "seconds", round(record["elapsed_seconds"], 1),
              flush=True)
    report = {
        "task": "unconditional next-sample waveform generation at 8 kHz, not text-to-speech",
        "source_manifest_sha256": sha256(manifest_path),
        "code_sha256": sha256(Path(__file__)),
        "speaker": manifest["speaker"],
        "train_chapters": manifest["training_chapters"],
        "validation_chapters": manifest["validation_chapters"],
        "train_sequences": len(train_codes), "validation_sequences": len(val_codes),
        "sequence_samples": 512,
        "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": "256-code embedding64, two-layer GRU width160, 256-way next-sample softmax",
        "device": str(device), "torch_version": torch.__version__,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "sample_rate": 8000,
        "uniform_nll_per_sample": float(np.log(256)),
        "train_unigram_validation_nll_per_sample": unigram_nll,
        "initial_validation_nll_per_sample": initial,
        "best_epoch_by_validation_nll": best_epoch,
        "best_validation_nll_per_sample": best,
        "best_checkpoint_sha256": sha256(out / "best.pt"),
        "epochs": epochs, "batch_size": batch_size,
        "history": history,
        "test_split_loaded": False,
        "wall_seconds": time.perf_counter() - start_time,
    }
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")


@torch.no_grad()
def generate(model: SampleGRU, prefix: np.ndarray, count: int,
             temperature: float, top_k: int, seed: int) -> np.ndarray:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = None
    last_logits = None
    for value in prefix:
        last_logits, state = model(torch.tensor([[int(value)]], dtype=torch.long), state)
    assert last_logits is not None
    output = []
    for i in range(count):
        logits = last_logits[0, -1] / temperature
        if top_k < 256:
            threshold = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probabilities = torch.softmax(logits, dim=0)
        next_code = int(torch.multinomial(probabilities, 1, generator=generator))
        output.append(next_code)
        last_logits, state = model(torch.tensor([[next_code]], dtype=torch.long), state)
        if (i + 1) % 2000 == 0:
            print("generated", i + 1, "/", count, flush=True)
    return np.array(output, dtype=np.uint8)


def sample(out: Path, run: Path, seconds: float) -> None:
    if out.exists():
        raise FileExistsError(f"preserve existing generated samples: {out}")
    source = json.loads((run / "train.json").read_text(encoding="utf-8"))
    manifest_sha = sha256(DATA / "manifest.json")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if (checkpoint["source_manifest_sha256"] != manifest_sha or
            source["best_checkpoint_sha256"] != sha256(run / "best.pt")):
        raise RuntimeError("model/data receipt mismatch")
    torch.set_num_threads(1)
    model = SampleGRU().cpu()
    model.load_state_dict(checkpoint["model"])
    train_codes = np.load(DATA / "train.npy", allow_pickle=False)
    # The prefix is a real TRAINING chunk, so its first 128 samples are known;
    # every subsequent sample is selected by this model, never copied from it.
    prefix = train_codes[0, :128]
    count = int(round(seconds * 8000))
    out.mkdir(parents=True)
    start_time = time.perf_counter()
    continuation_codes = generate(model, prefix, count, 0.9, 64, 20260924)
    continuation_seconds = time.perf_counter() - start_time
    start_time = time.perf_counter()
    unconditional_codes = generate(model, np.array([128], dtype=np.uint8),
                                   count, 0.9, 64, 7)
    unconditional_seconds = time.perf_counter() - start_time
    sf.write(out / "real_prefix_then_model.wav",
             decode_mulaw(np.concatenate([prefix, continuation_codes])), 8000)
    sf.write(out / "model_only_from_silence.wav", decode_mulaw(unconditional_codes), 8000)
    report = {
        "task": "sample-level audio generation, no text or phoneme condition",
        "source_run_train_sha256": sha256(run / "train.json"),
        "checkpoint_sha256": sha256(run / "best.pt"),
        "first_128_samples_source": "first training chunk of speaker 1272; not a held-out clip",
        "continuation_generated_samples": count,
        "unconditional_generated_samples": count,
        "sample_rate": 8000,
        "sampling_temperature": 0.9,
        "top_k": 64,
        "continuation_wall_seconds": continuation_seconds,
        "unconditional_wall_seconds": unconditional_seconds,
        "continuation_rms": float(np.sqrt(np.mean(decode_mulaw(continuation_codes) ** 2))),
        "unconditional_rms": float(np.sqrt(np.mean(decode_mulaw(unconditional_codes) ** 2))),
        "continuation_distinct_codes": len(np.unique(continuation_codes)),
        "unconditional_distinct_codes": len(np.unique(unconditional_codes)),
        "no_semantic_intelligibility_claim": True,
    }
    (out / "sample.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--out-dir", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=8)
    training.add_argument("--batch-size", type=int, default=32)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--run-dir", type=Path, required=True)
    sampling.add_argument("--out-dir", type=Path, required=True)
    sampling.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "train":
        out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
        if out.exists():
            raise FileExistsError(f"preserve existing waveform run: {out}")
        out.mkdir(parents=True)
        np.random.seed(20260924)
        torch.manual_seed(20260924)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260924)
        if sys.platform == "win32":
            torch.backends.cudnn.enabled = False
        torch.set_num_threads(4)
        train(out, args.epochs, args.batch_size)
    else:
        run = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
        out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
        sample(out, run, args.seconds)


if __name__ == "__main__":
    main()
