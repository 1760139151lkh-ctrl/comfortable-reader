"""Fetch a fixed, external Piper TTS voice for an explicitly separate comparison."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/data/piper_kristin_c10ece1"
REVISION = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
DIRECTORY = "en/en_US/kristin/medium"
MODEL_FILE = "en_US-kristin-medium.onnx"
EXPECTED_MODEL_SHA256 = "5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4"
EXPECTED_MODEL_BYTES = 63531379


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str) -> Path:
    target = OUT / name
    if target.exists():
        raise FileExistsError(f"preserve existing external model source: {target}")
    partial = OUT / (name + ".part")
    if partial.exists():
        raise FileExistsError(f"inspect incomplete download: {partial}")
    url = f"https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}/{DIRECTORY}/{name}"
    with urllib.request.urlopen(url, timeout=180) as source, partial.open("wb") as output:
        while chunk := source.read(1 << 20):
            output.write(chunk)
    partial.replace(target)
    return target


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"preserve existing voice source: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    model = fetch(MODEL_FILE)
    if model.stat().st_size != EXPECTED_MODEL_BYTES or sha256(model) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("voice model differs from pinned Hub LFS metadata")
    config = fetch(MODEL_FILE + ".json")
    card = fetch("MODEL_CARD")
    config_data = json.loads(config.read_text(encoding="utf-8"))
    if config_data["audio"]["sample_rate"] != 22050:
        raise RuntimeError("voice config sample rate changed")
    manifest = {
        "repository": "https://huggingface.co/rhasspy/piper-voices",
        "revision": REVISION,
        "directory": DIRECTORY,
        "model_sha256": sha256(model),
        "model_bytes": model.stat().st_size,
        "config_sha256": sha256(config),
        "model_card_sha256": sha256(card),
        "sample_rate": 22050,
        "identity": "external pretrained Piper English TTS voice; not trained by this book",
        "card_training_claim": "trained from scratch on about 11.5 hours of LibriVox recordings, per pinned MODEL_CARD",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"model_bytes": manifest["model_bytes"],
                      "model_sha256": manifest["model_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
