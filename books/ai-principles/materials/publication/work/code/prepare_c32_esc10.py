"""Resample fixed ESC-10 WAV clips with antialiasing and cache log-mel features."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from c32_audio_features import (FFT_SIZE, HOP_SAMPLES, MEL_BANDS,
                                SAMPLE_RATE, WINDOW_SAMPLES, log_mel)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/esc10_33c8ce9"
EXTRACTED = DATA / "extracted"
OUT = DATA / "chapter_features"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"preserve existing ESC-10 feature directory: {OUT}")
    source = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if sha256(DATA / "ESC-50-fixed-commit.zip") != source["archive_sha256"]:
        raise RuntimeError("ESC source archive changed")
    all_rows = list(csv.DictReader((EXTRACTED / "meta/esc50.csv").open(encoding="utf-8")))
    rows = [row for row in all_rows if row["esc10"].lower() == "true"]
    if len(rows) != 400:
        raise RuntimeError("ESC-10 metadata count changed")
    OUT.mkdir(parents=True)
    features = OUT / "logmel80"
    features.mkdir()
    chapter_rows = []
    for row in rows:
        path = EXTRACTED / "audio" / row["filename"]
        wave, rate = sf.read(path, dtype="float32")
        if rate != 44100 or wave.ndim != 1 or len(wave) != 220500:
            raise ValueError(f"unexpected ESC audio: {path}")
        down = resample_poly(wave, 160, 441).astype(np.float32)
        if len(down) != 80000:
            raise ValueError(f"resampled length mismatch: {path}")
        mel = log_mel(down)
        np.save(features / (path.stem + ".npy"), mel.astype(np.float16),
                allow_pickle=False)
        fold = int(row["fold"])
        chapter_rows.append({
            "id": path.stem,
            "source_wav": path.relative_to(ROOT).as_posix(),
            "source_recording": row["src_file"],
            "fold": fold,
            "split": "train" if fold <= 3 else "validation" if fold == 4 else "test",
            "category": row["category"],
            "target": int(row["target"]),
            "frames": int(mel.shape[0]),
        })
    source_folds = {}
    for row in chapter_rows:
        source_folds.setdefault(row["source_recording"], set()).add(row["fold"])
    if any(len(folds) > 1 for folds in source_folds.values()):
        raise RuntimeError("same source recording crosses folds")
    manifest = {
        "identity": "author one-fold ESC-10 experiment: folds 1-3 train, fold 4 validation, fold 5 test; not upstream five-fold mean",
        "upstream_archive_sha256": source["archive_sha256"],
        "upstream_commit": source["commit"],
        "source_csv_sha256": sha256(EXTRACTED / "meta/esc50.csv"),
        "source_license_sha256": sha256(EXTRACTED / "LICENSE"),
        "class_names": sorted({row["category"] for row in chapter_rows}),
        "split_counts": {split: sum(row["split"] == split for row in chapter_rows)
                         for split in ("train", "validation", "test")},
        "sample_rate_original": 44100,
        "sample_rate_features": SAMPLE_RATE,
        "resampling": "scipy.signal.resample_poly(up=160, down=441), includes FIR anti-alias filter",
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "fft_size": FFT_SIZE,
        "mel_bands": MEL_BANDS,
        "rows": chapter_rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"split_counts": manifest["split_counts"],
                      "manifest_sha256": sha256(OUT / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
