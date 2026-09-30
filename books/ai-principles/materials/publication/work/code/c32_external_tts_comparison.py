"""Run a pinned external Piper TTS voice locally, keeping its identity separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import piper
from piper import PiperVoice


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/data/piper_kristin_c10ece1"
MODEL = SOURCE / "en_US-kristin-medium.onnx"
EXAMPLES = [
    ("dog_and_rain", "The dog barked once, and then the rain began."),
    ("train_arrives", "Please wait. The train is coming."),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing external TTS comparison: {out}")
    source = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if source["model_sha256"] != sha256(MODEL):
        raise RuntimeError("pinned voice model changed")
    out.mkdir(parents=True)
    # The Piper 1.4.2 Windows wheel's espeak bridge did not resolve the
    # bundled data through this Chinese-named workspace path. Give it an
    # explicit ASCII temp path containing the same bundled data instead.
    bundled_espeak = Path(piper.__file__).resolve().parent / "espeak-ng-data"
    espeak_data = Path(tempfile.gettempdir()) / "c32-piper-espeak-20260924"
    if espeak_data.exists():
        if not espeak_data.is_dir():
            raise RuntimeError("espeak data cache path is not a directory")
    else:
        shutil.copytree(bundled_espeak, espeak_data)
    for name in ("phontab", "en_dict"):
        if sha256(espeak_data / name) != sha256(bundled_espeak / name):
            raise RuntimeError(f"espeak data cache differs: {name}")
    load_start = time.perf_counter()
    voice = PiperVoice.load(MODEL, use_cuda=False,
                            espeak_data_dir=espeak_data)
    load_seconds = time.perf_counter() - load_start
    results = []
    for name, sentence in EXAMPLES:
        path = out / (name + ".wav")
        start = time.perf_counter()
        with wave.open(str(path), "wb") as output:
            voice.synthesize_wav(sentence, output)
        seconds = time.perf_counter() - start
        info = sf.info(path)
        audio, rate = sf.read(path, dtype="float32")
        if rate != 22050 or audio.ndim != 1 or len(audio) < rate:
            raise RuntimeError(f"invalid synthesized audio: {path}")
        results.append({
            "sentence": sentence,
            "file": path.relative_to(ROOT).as_posix(),
            "file_sha256": sha256(path),
            "sample_rate": rate,
            "samples": len(audio),
            "duration_seconds": info.duration,
            "inference_wall_seconds": seconds,
            "rms": float(np.sqrt(np.mean(audio ** 2))),
            "peak_abs": float(np.max(np.abs(audio))),
        })
    report = {
        "identity": "external pinned Piper TTS inference, not this book's trained waveform model",
        "voice_revision": source["revision"],
        "voice_model_sha256": source["model_sha256"],
        "voice_model_card_sha256": source["model_card_sha256"],
        "voice_config_sha256": source["config_sha256"],
        "voice_dataset_and_training_claim": source["card_training_claim"],
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "code_sha256": sha256(Path(__file__)),
        "model_load_seconds": load_seconds,
        "espeak_data_cache": str(espeak_data),
        "espeak_phontab_sha256": sha256(espeak_data / "phontab"),
        "external_model_parameters_updated": False,
        "examples": results,
        "no_human_intelligibility_or_preference_rating": True,
    }
    (out / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"voice": source["revision"],
                      "outputs": [(r["samples"], r["duration_seconds"]) for r in results]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
