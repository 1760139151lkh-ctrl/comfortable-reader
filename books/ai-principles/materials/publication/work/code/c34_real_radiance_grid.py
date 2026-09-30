"""Train a small view-independent 3D radiance grid from random weights on ETH3D pipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/eth3d_pipes/views_518"
VAL_INDEX = 3   # DSC_0637: chosen by source index before training
TEST_INDEX = 10  # DSC_0644: pixels opened only after validation choice
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RadianceGrid(nn.Module):
    def __init__(self, side: int, minimum: np.ndarray,
                 maximum: np.ndarray, samples: int) -> None:
        super().__init__()
        initial = torch.randn(1, 4, side, side, side) * 0.03
        initial[:, 0] -= 1.0  # finite starting opacity, not a pretrained shape
        self.values = nn.Parameter(initial)
        self.register_buffer("minimum", torch.tensor(minimum, dtype=torch.float32))
        self.register_buffer("maximum", torch.tensor(maximum, dtype=torch.float32))
        self.register_buffer("distances", torch.linspace(0.15, 8.0, samples))

    def forward(self, origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        rays = origins.shape[0]
        positions = origins[:, None, :] + directions[:, None, :] * self.distances[None, :, None]
        normalized = 2.0 * (positions - self.minimum) / (self.maximum - self.minimum) - 1.0
        inside = ((normalized >= -1) & (normalized <= 1)).all(dim=-1)
        # 5D grid_sample: x goes along W, y along H, z along D.
        queried = F.grid_sample(self.values, normalized.reshape(1, rays * len(self.distances), 1, 1, 3),
                                mode="bilinear", padding_mode="zeros", align_corners=True)
        features = queried.reshape(4, rays, len(self.distances)).permute(1, 2, 0)
        density = F.softplus(features[:, :, 0]) * inside
        colors = torch.sigmoid(features[:, :, 1:4])
        delta = torch.diff(self.distances, append=self.distances[-1:] +
                           (self.distances[-1] - self.distances[-2]))
        opacity = 1.0 - torch.exp(-density * delta[None, :])
        survival = torch.cumprod(torch.cat([torch.ones_like(opacity[:, :1]),
                                             1.0 - opacity + 1e-10], dim=1), dim=1)[:, :-1]
        weights = survival * opacity
        return ((weights[:, :, None] * colors).sum(dim=1) +
                (1.0 - weights.sum(dim=1, keepdim=True)).clamp(0, 1))


def photo(view: dict, width: int) -> np.ndarray:
    path = ROOT / view["small_path"]
    if sha256(path) != view["small_sha256"]:
        raise RuntimeError(f"view bytes changed: {path}")
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    height = round(bgr.shape[0] * width / bgr.shape[1])
    small = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def intrinsics(original: dict, width: int, height: int) -> dict:
    return {"fx": original["fx"] * width / original["width"],
            "fy": original["fy"] * height / original["height"],
            "cx": original["cx"] * width / original["width"],
            "cy": original["cy"] * height / original["height"]}


def rays(view: dict, coordinates: torch.Tensor, camera: dict,
         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    u = coordinates[:, 0].float() + 0.5
    v = coordinates[:, 1].float() + 0.5
    local = torch.stack([(u - camera["cx"]) / camera["fx"],
                         (v - camera["cy"]) / camera["fy"], torch.ones_like(u)], dim=-1)
    local = F.normalize(local, dim=-1)
    R = torch.tensor(view["rotation_world_to_camera"], dtype=torch.float32, device=device)
    center = torch.tensor(view["center_world"], dtype=torch.float32, device=device)
    directions = local @ R
    return center.expand_as(directions), F.normalize(directions, dim=-1)


@torch.no_grad()
def evaluate_pixels(model: RadianceGrid, view: dict, image: np.ndarray,
                    camera: dict, pixel_index: np.ndarray,
                    device: torch.device, chunk: int = 2048) -> float:
    height, width = image.shape[:2]
    truth = torch.from_numpy(image.reshape(-1, 3)).to(device)
    total = 0.0
    for start in range(0, len(pixel_index), chunk):
        ids = pixel_index[start:start + chunk]
        xy = torch.tensor(np.stack([ids % width, ids // width], axis=1),
                          dtype=torch.long, device=device)
        o, d = rays(view, xy, camera, device)
        pred = model(o, d)
        total += (pred - truth[ids]).square().sum().item()
    return total / (len(pixel_index) * 3)


@torch.no_grad()
def render_image(model: RadianceGrid, view: dict, camera: dict,
                 height: int, width: int, device: torch.device,
                 chunk: int = 2048) -> np.ndarray:
    result = np.empty((height * width, 3), dtype=np.float32)
    for start in range(0, len(result), chunk):
        ids = np.arange(start, min(start + chunk, len(result)))
        xy = torch.tensor(np.stack([ids % width, ids // width], axis=1),
                          dtype=torch.long, device=device)
        o, d = rays(view, xy, camera, device)
        result[start:start + len(ids)] = model(o, d).cpu().numpy()
    return result.reshape(height, width, 3)


def plot(out: Path, rows: list[tuple[str, np.ndarray, np.ndarray,
                                     np.ndarray]], width: int) -> str:
    figure, axes = plt.subplots(len(rows), 4, figsize=(14, 3.2 * len(rows)), layout="constrained")
    for ax_row, (name, real, nearest, generated) in zip(axes, rows):
        error = np.abs(generated - real).mean(axis=2)
        for ax, arr, title in zip(ax_row,
                                  [real, nearest, generated, error],
                                  ["真实留出视角", "最近相机的原训练图", "随机参数训练后的三维场", "绝对颜色误差"]):
            ax.imshow(arr, cmap="magma" if arr.ndim == 2 else None,
                      vmin=0, vmax=0.35 if arr.ndim == 2 else 1)
            ax.set_title(f"{name}：{title}")
            ax.set_xticks([])
            ax.set_yticks([])
    figure.suptitle(f"ETH3D pipes：{width}像素宽的真实多视角学习与留出视角", fontsize=15)
    path = out / "heldout_views.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-rays", type=int, default=1024)
    parser.add_argument("--width", type=int, default=260)
    parser.add_argument("--grid-side", type=int, default=96)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--smoke", action="store_true", help="check training/validation path without opening held-out test")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.width < 64 or args.samples < 4:
        raise ValueError("invalid training settings")
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(out)
    manifest_path = DATA / "manifest.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = source["views"]
    if len(views) != 14:
        raise RuntimeError("expected 14 actual cameras")
    train_indices = [i for i in range(14) if i not in (VAL_INDEX, TEST_INDEX)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(20260924)
    np.random.seed(20260924)
    torch.manual_seed(20260924)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260924)
        torch.cuda.reset_peak_memory_stats()
    # Deliberately DO NOT open TEST_INDEX image before model selection.
    train_images = [photo(views[i], args.width) for i in train_indices]
    val_image = photo(views[VAL_INDEX], args.width)
    height, width = val_image.shape[:2]
    if any(img.shape[:2] != (height, width) for img in train_images):
        raise RuntimeError("all view image shapes must agree")
    cam = intrinsics(source["small_intrinsics"], width, height)
    train_target = torch.tensor(np.stack(train_images), dtype=torch.float32, device=device)
    all_xy = torch.tensor(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).reshape(-1, 2),
                          dtype=torch.long, device=device)
    origin_cache = []
    direction_cache = []
    for index in train_indices:
        o, d = rays(views[index], all_xy, cam, device)
        origin_cache.append(o.reshape(height, width, 3))
        direction_cache.append(d.reshape(height, width, 3))
    origin_cache = torch.stack(origin_cache)
    direction_cache = torch.stack(direction_cache)
    point_bounds = source["sfm_point_percentile_bounds"]
    minimum = np.asarray(point_bounds["min"]) - np.array([0.4, 0.3, 0.3])
    maximum = np.asarray(point_bounds["max"]) + np.array([0.4, 0.3, 0.3])
    model = RadianceGrid(args.grid_side, minimum, maximum, args.samples).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    fixed_val = np.random.default_rng(541).choice(height * width,
                                                  size=min(4096, height * width), replace=False)
    initial_val = evaluate_pixels(model, views[VAL_INDEX], val_image,
                                  cam, fixed_val, device)
    best_val, best_step = initial_val, 0
    out.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 0}, out / "best.pt")
    history = []
    start_clock = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        chosen = np.random.randint(0, len(train_indices), size=args.batch_rays)
        xy_np = np.column_stack((np.random.randint(0, width, size=args.batch_rays),
                                 np.random.randint(0, height, size=args.batch_rays)))
        chosen = torch.tensor(chosen, dtype=torch.long, device=device)
        xy = torch.tensor(xy_np, dtype=torch.long, device=device)
        origins = origin_cache[chosen, xy[:, 1], xy[:, 0]]
        directions = direction_cache[chosen, xy[:, 1], xy[:, 0]]
        target = train_target[chosen, xy[:, 1], xy[:, 0]]
        pred = model(origins, directions)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            model.eval()
            val_mse = evaluate_pixels(model, views[VAL_INDEX], val_image,
                                      cam, fixed_val, device)
            entry = {"step": step, "train_batch_mse": float(loss.item()),
                     "validation_fixed_rays_mse": val_mse}
            history.append(entry)
            if val_mse < best_val:
                best_val, best_step = val_mse, step
                torch.save({"model": model.state_dict(), "step": step}, out / "best.pt")
            print(entry, flush=True)
    if args.smoke:
        receipt = {"identity": "C34 implementation smoke; train/validation only, held-out test pixels not opened",
                   "steps": args.steps, "initial_val_mse": initial_val,
                   "best_val_mse": best_val, "best_step": best_step,
                   "history": history, "device": str(device)}
        (out / "smoke.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print("smoke validation only", receipt["best_val_mse"], flush=True)
        return
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # Test pixels first opened after all training/validation choices were fixed.
    test_image = photo(views[TEST_INDEX], args.width)
    if test_image.shape[:2] != (height, width):
        raise RuntimeError("held-out test view shape mismatch")
    full_ids = np.arange(height * width)
    val_full_mse = evaluate_pixels(model, views[VAL_INDEX], val_image,
                                   cam, full_ids, device)
    test_full_mse = evaluate_pixels(model, views[TEST_INDEX], test_image,
                                    cam, full_ids, device)
    val_render = render_image(model, views[VAL_INDEX], cam, height, width, device)
    test_render = render_image(model, views[TEST_INDEX], cam, height, width, device)
    centers = np.asarray([v["center_world"] for v in views])
    closest = {}
    comparisons = []
    for index, real, generated in ((VAL_INDEX, val_image, val_render),
                                   (TEST_INDEX, test_image, test_render)):
        nearest_index = min(train_indices, key=lambda i: np.linalg.norm(centers[i] - centers[index]))
        nearest = train_images[train_indices.index(nearest_index)]
        nearest_mse = float(np.mean((nearest - real) ** 2))
        closest[str(index)] = {"train_view_index": nearest_index,
                               "same_pixel_copy_mse": nearest_mse,
                               "camera_center_distance": float(np.linalg.norm(centers[nearest_index] - centers[index]))}
        comparisons.append((views[index]["name"], real, nearest, generated))
    figure_sha = plot(out, comparisons, width)
    np.savez_compressed(out / "heldout_renders.npz", val=val_render,
                        test=test_render, val_real=val_image, test_real=test_image)
    report = {
        "identity": "author small view-independent differentiable 3D radiance GRID from random parameters on ETH3D real pipes; not original MLP NeRF, not laser scan mesh",
        "source_manifest_sha256": sha256(manifest_path),
        "train_view_indices": train_indices, "validation_view_index": VAL_INDEX,
        "test_view_index": TEST_INDEX,
        "test_image_pixels_opened_only_after_best_validation_step": True,
        "caveat": "all 14 source images were previously viewed by the author in an orientation montage; test is model-held-out, not wholly unseen source to the author",
        "camera_intrinsics_at_training_resolution": cam,
        "world_aabb": {"min": minimum.tolist(), "max": maximum.tolist()},
        "grid_side": args.grid_side, "ray_samples": args.samples,
        "near_far_world_distance": [0.15, 8.0],
        "model_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device), "steps_planned_and_completed": args.steps,
        "initial_validation_fixed_rays_mse": initial_val,
        "best_step_by_validation_fixed_rays": best_step,
        "best_validation_fixed_rays_mse": best_val,
        "validation_full_mse": val_full_mse,
        "test_full_mse": test_full_mse,
        "validation_psnr_db": -10.0 * math.log10(val_full_mse),
        "test_psnr_db": -10.0 * math.log10(test_full_mse),
        "nearest_camera_copy": closest,
        "history": history,
        "wall_seconds": time.perf_counter() - start_clock,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "best_checkpoint_sha256": sha256(out / "best.pt"),
        "heldout_renders_sha256": sha256(out / "heldout_renders.npz"),
        "figure_sha256": figure_sha,
        "limits": "view interpolation/extrapolation only on this captured static scene and known camera poses; grid is fixed-resolution, view-independent RGB and does not yield editable mesh or physical material parameters; no official ETH3D novel-view score claimed",
    }
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps({"best_step": best_step, "val_psnr": report["validation_psnr_db"],
                      "test_psnr": report["test_psnr_db"], "test_nearest_copy_mse": closest[str(TEST_INDEX)]["same_pixel_copy_mse"]}), flush=True)


if __name__ == "__main__":
    main()
