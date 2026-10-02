"""Show COCO image 724's real instance boxes and one polygon mask without training a detector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "work/data/coco2017_val_c35/images/000000000724.jpg"
IMAGE_MANIFEST = ROOT / "work/data/coco2017_val_c35/extraction_manifest.json"
INSTANCE = ROOT / "work/data/c35_object_presence/instances_val2017.json"
INSTANCE_MANIFEST = ROOT / "work/data/c35_object_presence/manifest.json"
ARCHIVE = ROOT / "work/data/coco2017_val_c35/annotations_trainval2017.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap_w = max(0., min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0., min(ay + ah, by + bh) - max(ay, by))
    intersection = overlap_w * overlap_h
    return intersection / (aw * ah + bw * bh - intersection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    dest = (ROOT / args.out_dir).resolve()
    if not dest.is_relative_to(ROOT / "work") or (dest.exists() and any(dest.iterdir())):
        raise ValueError("Choose a new empty output directory under package/work")
    dest.mkdir(parents=True, exist_ok=True)
    original = json.loads(IMAGE_MANIFEST.read_text(encoding="utf-8"))
    instance_info = json.loads(INSTANCE_MANIFEST.read_text(encoding="utf-8"))
    if sha(ARCHIVE) != instance_info["source_zip_sha256"] or sha(INSTANCE) != instance_info["instance_json_sha256"]:
        raise RuntimeError("COCO original annotation identity changed")
    images = original["extracted"]["files"]
    if sha(IMAGE) != images[IMAGE.name]["sha256"]:
        raise RuntimeError("COCO original JPEG changed")
    source = json.loads(INSTANCE.read_text(encoding="utf-8"))
    names = {item["id"]: item["name"] for item in source["categories"]}
    signs = [item for item in source["annotations"]
             if item["image_id"] == 724 and names[item["category_id"]] == "stop sign"]
    signs.sort(key=lambda row: row["area"], reverse=True)
    if len(signs) != 2 or signs[0]["iscrowd"] or not isinstance(signs[0]["segmentation"], list):
        raise RuntimeError("Expected two polygon-annotated stop-sign instances")
    picture = Image.open(IMAGE).convert("RGB")
    mask = Image.new("L", picture.size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in signs[0]["segmentation"]:
        xy = list(zip(polygon[0::2], polygon[1::2]))
        draw.polygon(xy, fill=255)
    mask_array = np.asarray(mask) > 0
    x, y, w, h = signs[0]["bbox"]
    shifted = [x + 20, y, w, h]
    iou = box_iou(signs[0]["bbox"], shifted)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.6), constrained_layout=True)
    for axis in axes:
        axis.imshow(picture)
        axis.set_axis_off()
    axes[0].set_title("同一张原照片")
    for sign, color in zip(signs, ("#e44035", "#f5aa2a")):
        xx, yy, ww, hh = sign["bbox"]
        axes[1].add_patch(Rectangle((xx, yy), ww, hh, fill=False, ec=color, lw=2.2))
    axes[1].set_title("两个路牌实例的框")
    overlay = np.zeros((picture.height, picture.width, 4), dtype=float)
    overlay[..., :3] = (0.1, 0.84, 0.8)
    overlay[..., 3] = mask_array * .45
    axes[2].imshow(overlay)
    axes[2].add_patch(Rectangle((x, y), w, h, fill=False, ec="#e44035", lw=1.5))
    axes[2].set_title("大路牌的像素区域与外框")
    plot_path = dest / "coco724_box_and_mask.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    report = {
        "scope": "One original COCO 2017 validation image and human instance annotations; no learned detection/segmentation or official benchmark test",
        "image_id": 724, "source_image_sha256": sha(IMAGE), "source_instance_json_sha256": sha(INSTANCE),
        "image_width_height": list(picture.size), "stop_sign_instances": len(signs),
        "large_box_xywh": signs[0]["bbox"], "large_box_area": w * h,
        "large_annotation_polygon_area": signs[0]["area"],
        "large_rasterized_mask_pixels": int(mask_array.sum()),
        "small_box_xywh": signs[1]["bbox"],
        "author_horizontal_shift_px": 20,
        "large_box_vs_shifted_box_iou": iou,
        "figure_sha256": sha(plot_path),
    }
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
