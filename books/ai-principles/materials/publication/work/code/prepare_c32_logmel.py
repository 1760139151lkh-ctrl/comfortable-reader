"""Cache real LibriSpeech log-mel features, with split/source receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from c32_audio_features import (FFT_SIZE, HOP_SAMPLES, MEL_BANDS,
                                SAMPLE_RATE, WINDOW_SAMPLES, log_mel)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/librispeech_dev_clean"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="0 extracts every clip; positive limit is only a smoke run")
    parser.add_argument("--out-dir", type=Path, default=DATA / "features/logmel80")
    args = parser.parse_args()
    directory = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"preserve existing features: {directory}")
    split_path = DATA / "chapter_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    rows = split["rows"] if args.limit == 0 else split["rows"][:args.limit]
    directory.mkdir(parents=True, exist_ok=True)
    shapes = []
    for number, row in enumerate(rows, 1):
        wave, rate = sf.read(ROOT / row["flac"], dtype="float32")
        if rate != SAMPLE_RATE or wave.ndim != 1 or len(wave) != row["samples"]:
            raise ValueError(f"audio changed: {row['id']}")
        features = log_mel(wave)
        if features.shape[1] != MEL_BANDS or not np.isfinite(features).all():
            raise ValueError(f"invalid features: {row['id']}")
        np.save(directory / f"{row['id']}.npy", features.astype(np.float16),
                allow_pickle=False)
        shapes.append([row["id"], int(features.shape[0])])
        if number % 250 == 0:
            print(f"features {number}/{len(rows)}", flush=True)
    receipt = {
        "source_archive_sha256": split["archive_sha256"],
        "chapter_split_sha256": sha256(split_path),
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "fft_size": FFT_SIZE,
        "mel_bands": MEL_BANDS,
        "mel_formula": "2595 log10(1+Hz/700); triangle, normalized to unit sum over FFT bins",
        "power_log_floor": 1e-10,
        "split_role": "feature cache includes every split; training code must load only train/validation until model frozen",
        "cached_count": len(rows),
        "total_frames": sum(shape[1] for shape in shapes),
        "frames_by_id": shapes,
    }
    (directory / "manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cached_count": len(rows),
                      "total_frames": receipt["total_frames"],
                      "feature_manifest_sha256": sha256(directory / "manifest.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
