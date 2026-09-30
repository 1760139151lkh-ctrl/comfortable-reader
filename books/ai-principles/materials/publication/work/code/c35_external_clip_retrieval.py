"""Read-only inference with OpenAI's published 2021 CLIP ViT-B/32 weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import clip
from clip.clip import _MODELS
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from c35_image_text_retrieval import DATA

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SOURCE = ROOT / "data" / "coco2017_val_c35" / "images"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(scores: torch.Tensor) -> dict:
    result = {}
    n = scores.shape[0]
    for direction, matrix in (("image_to_text", scores), ("text_to_image", scores.T)):
        rank = ((matrix.argsort(dim=1, descending=True) == torch.arange(n)[:, None]).nonzero()[:, 1] + 1).float()
        result[direction] = {"r1": float((rank <= 1).float().mean()), "r5": float((rank <= 5).float().mean()), "r10": float((rank <= 10).float().mean()), "median_rank": float(rank.median()), "mean_rank": float(rank.mean())}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", type=Path, default=Path("work/data/c35_external_openai_clip"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError("existing result")
    a.weights_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = json.loads((DATA / "image_records.json").read_text(encoding="utf-8"))
    test = [r for r in rows if r["split"] == "test"]
    if len(test) != 500:
        raise RuntimeError("test size changed")
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(a.weights_dir), jit=False)
    model.eval()
    weight_files = list(a.weights_dir.glob("*.pt"))
    if len(weight_files) != 1:
        raise RuntimeError(f"unexpected weight files: {weight_files}")
    images, texts = [], []
    with torch.inference_mode():
        for start in range(0, len(test), 32):
            batch = test[start:start + 32]
            pixel_tensors = []
            for row in batch:
                with Image.open(IMAGE_SOURCE / row["source_file"]) as image:
                    pixel_tensors.append(preprocess(image.convert("RGB")))
            pixels = torch.stack(pixel_tensors).to(device)
            words = clip.tokenize([row["captions"][0] for row in batch], truncate=True).to(device)
            images.append(F.normalize(model.encode_image(pixels).float(), dim=-1).cpu())
            texts.append(F.normalize(model.encode_text(words).float(), dim=-1).cpu())
            print(f"encoded {min(start + 32, len(test))}", flush=True)
    sim = torch.cat(images) @ torch.cat(texts).T
    output = {
        "identity": "OpenAI published CLIP ViT-B/32 pretrained weights, local inference only; NO gradient or new training",
        "official_code": "https://github.com/openai/CLIP",
        "official_code_commit_installed": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
        "official_paper": "https://arxiv.org/abs/2103.00020",
        "weight_file": str(weight_files[0]),
        "weight_bytes": weight_files[0].stat().st_size,
        "weight_sha256": sha(weight_files[0]),
        "weight_url": _MODELS["ViT-B/32"],
        "test": "same 500 author-internal COCO2017 val image IDs and first human caption; original full-res images with official CLIP preprocessing/tokenizer, unlike C35 local 112px/30-token tiny model",
        "external_training_overlap_with_these_500_coco_images": "UNKNOWN; original CLIP learned on external internet image-text pairs and local COCO image hashes were not compared with that inaccessible training corpus",
        "scores": summarize(sim),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["scores"], ensure_ascii=False))


if __name__ == "__main__":
    main()
