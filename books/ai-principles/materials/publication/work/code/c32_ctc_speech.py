"""Train a small character CTC recognizer from random weights on real speech.

Official LibriSpeech dev-clean is re-split by speaker for this book. This is
not an official LibriSpeech benchmark run. The test split is never read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from c32_audio_features import MEL_BANDS, normalize_utterance


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/librispeech_dev_clean"
SPLIT = DATA / "chapter_split.json"
FEATURES = DATA / "features/logmel80"
ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
CHAR_TO_ID = {char: i + 1 for i, char in enumerate(ALPHABET)}
ID_TO_CHAR = {i: char for char, i in CHAR_TO_ID.items()}
BLANK = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1,
                               previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def collapse(path: list[int]) -> str:
    previous = None
    result = []
    for label in path:
        if label != previous and label != BLANK:
            result.append(ID_TO_CHAR[label])
        previous = label
    return " ".join("".join(result).split())


def output_length(input_length: int) -> int:
    return (input_length + 1) // 2


class CTCRecognizer(nn.Module):
    def __init__(self, width: int = 192, layers: int = 3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(MEL_BANDS, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.rnn = nn.GRU(width, width, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(2 * width, len(ALPHABET) + 1)

    def forward(self, features: torch.Tensor,
                lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(features.transpose(1, 2)).transpose(1, 2)
        result_lengths = (lengths + 1) // 2
        packed = nn.utils.rnn.pack_padded_sequence(
            x, result_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        return self.head(x), result_lengths


def batches(rows: list[dict], frames: dict[str, int], batch_size: int,
            rng: random.Random | None) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda row: frames[row["id"]])
    groups = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    if rng is not None:
        rng.shuffle(groups)
    return groups


def tensor_batch(rows: list[dict], device: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = []
    targets = []
    for row in rows:
        array = np.load(FEATURES / f"{row['id']}.npy", allow_pickle=False)
        array = normalize_utterance(array.astype(np.float32))
        arrays.append(array)
        targets.append([CHAR_TO_ID[char] for char in row["transcript"]])
    lengths = torch.tensor([len(array) for array in arrays], dtype=torch.long)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    padded = np.zeros((len(rows), max(lengths).item(), MEL_BANDS), dtype=np.float32)
    for i, array in enumerate(arrays):
        padded[i, :len(array)] = array
    return (torch.from_numpy(padded).to(device), lengths,
            torch.tensor([c for target in targets for c in target], dtype=torch.long),
            target_lengths)


@torch.no_grad()
def evaluate(model: CTCRecognizer, rows: list[dict],
             frames: dict[str, int], batch_size: int, device: torch.device
             ) -> dict:
    model.eval()
    objective = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=False)
    total_loss = 0.0
    characters = 0
    character_edits = 0
    words = 0
    word_edits = 0
    examples = []
    for group in batches(rows, frames, batch_size, None):
        x, lengths, targets, target_lengths = tensor_batch(group, device)
        logits, out_lengths = model(x, lengths)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        loss = objective(log_probs, targets, out_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite validation CTC loss")
        total_loss += loss.item() * len(group)
        predicted = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
        for row, path, length in zip(group, predicted, out_lengths.tolist()):
            hypothesis = collapse(path[:length])
            truth = row["transcript"]
            characters += len(truth)
            character_edits += edit_distance(hypothesis, truth)
            words += len(truth.split())
            word_edits += edit_distance(hypothesis.split(), truth.split())
            if len(examples) < 4:
                examples.append({"id": row["id"], "reference": truth,
                                 "hypothesis": hypothesis})
    return {"mean_ctc_loss": total_loss / len(rows),
            "character_error_rate": character_edits / characters,
            "word_error_rate": word_edits / words,
            "character_edits": character_edits, "reference_characters": characters,
            "word_edits": word_edits, "reference_words": words,
            "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-train", type=int, default=0,
                        help="positive number for a smoke run only")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="optional same-task model checkpoint; optimizer restarts")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing run: {out}")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    feature_manifest = json.loads((FEATURES / "manifest.json").read_text(encoding="utf-8"))
    if feature_manifest["chapter_split_sha256"] != sha256(SPLIT):
        raise RuntimeError("feature source/split changed")
    if feature_manifest["cached_count"] != len(split["rows"]):
        raise RuntimeError("feature cache incomplete")
    frames = dict(feature_manifest["frames_by_id"])
    train = [row for row in split["rows"] if row["split"] == "train"]
    validation = [row for row in split["rows"] if row["split"] == "validation"]
    if args.max_train:
        train = train[:args.max_train]
    if not train or not validation:
        raise ValueError("empty training or validation split")
    for row in train + validation:
        target = row["transcript"]
        required_frames = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        if output_length(frames[row["id"]]) < required_frames:
            raise ValueError(f"CTC path impossible: {row['id']}")
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if sys.platform == "win32":
        # On this host, repeated CUDA/cuDNN GRU use ended with native
        # 0xC0000409 after all files were written. Native CUDA RNN kernels
        # run to a clean process exit; keep the GPU for the other operations.
        torch.backends.cudnn.enabled = False
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTCRecognizer().to(device)
    init_checkpoint_sha = None
    if args.init_checkpoint is not None:
        initial_path = (args.init_checkpoint if args.init_checkpoint.is_absolute()
                        else ROOT / args.init_checkpoint)
        previous = torch.load(initial_path, map_location="cpu", weights_only=False)
        if (previous["chapter_split_sha256"] != sha256(SPLIT) or
                previous["feature_manifest_sha256"] != sha256(FEATURES / "manifest.json") or
                previous["alphabet"] != ALPHABET):
            raise RuntimeError("initial checkpoint does not match this task")
        model.load_state_dict(previous["model"])
        init_checkpoint_sha = sha256(initial_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=0.01)
    objective = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=False)
    rng = random.Random(args.seed)
    history = []
    run_start = time.perf_counter()
    initial = evaluate(model, validation, frames, args.batch_size, device)
    best = initial["mean_ctc_loss"]
    best_epoch = 0
    torch.save({"model": model.state_dict(), "epoch": 0,
                "alphabet": ALPHABET,
                "chapter_split_sha256": sha256(SPLIT),
                "feature_manifest_sha256": sha256(FEATURES / "manifest.json"),
                "width": 192, "layers": 3}, out / "best.pt")
    print("epoch 0 validation", initial["mean_ctc_loss"],
          initial["character_error_rate"], initial["word_error_rate"], flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        epoch_start = time.perf_counter()
        for group in batches(train, frames, args.batch_size, rng):
            x, lengths, targets, target_lengths = tensor_batch(group, device)
            logits, out_lengths = model(x, lengths)
            log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
            loss = objective(log_probs, targets, out_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training CTC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(group)
            total_seen += len(group)
        validation_result = evaluate(model, validation, frames,
                                     args.batch_size, device)
        record = {"epoch": epoch, "train_mean_ctc_loss": total_loss / total_seen,
                  "validation": validation_result,
                  "elapsed_seconds": time.perf_counter() - epoch_start}
        history.append(record)
        if validation_result["mean_ctc_loss"] < best:
            best = validation_result["mean_ctc_loss"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "alphabet": ALPHABET,
                        "chapter_split_sha256": sha256(SPLIT),
                        "feature_manifest_sha256": sha256(FEATURES / "manifest.json"),
                        "width": 192, "layers": 3}, out / "best.pt")
        print("epoch", epoch, "train", round(record["train_mean_ctc_loss"], 4),
              "val", round(validation_result["mean_ctc_loss"], 4),
              "CER", round(validation_result["character_error_rate"], 4),
              "WER", round(validation_result["word_error_rate"], 4),
              "seconds", round(record["elapsed_seconds"], 1), flush=True)
    report = {
        "task": "character CTC recognition from random weights on author speaker split of official LibriSpeech dev-clean",
        "official_benchmark": False,
        "source_archive_sha256": split["archive_sha256"],
        "chapter_split_sha256": sha256(SPLIT),
        "feature_manifest_sha256": sha256(FEATURES / "manifest.json"),
        "code_sha256": sha256(Path(__file__)),
        "init_checkpoint_sha256": init_checkpoint_sha,
        "optimizer_restarted_at_initial_checkpoint": init_checkpoint_sha is not None,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "train_clips": len(train), "validation_clips": len(validation),
        "test_clips_read": 0,
        "architecture": "Conv1d stride2 + Conv1d + 3-layer bidirectional GRU width192 + character head",
        "parameters": sum(p.numel() for p in model.parameters()),
        "alphabet": ALPHABET, "blank_id": BLANK,
        "ctc_reduction": "PyTorch mean: path NLL divided by target length, then batch mean",
        "decoding": "framewise argmax, collapse consecutive equal ids then remove blank; no language model",
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "initial_validation": initial,
        "best_epoch_by_validation_ctc_loss": best_epoch,
        "best_checkpoint_sha256": sha256(out / "best.pt"),
        "history": history,
        "wall_seconds": time.perf_counter() - run_start,
    }
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
