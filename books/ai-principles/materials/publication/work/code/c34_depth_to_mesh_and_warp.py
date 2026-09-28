"""Back-project real ETH3D scan disparity into a partial editable mesh and warp a view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from c34_real_stereo_baselines import estimate
from c34_stereo_pair_inspect import read_calib, read_gray, read_pfm


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/data/eth3d_two_view"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def forward_warp(left: np.ndarray, disparity: np.ndarray,
                 permitted: np.ndarray, focal: float, baseline_mm: float,
                 doffs: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = left.shape
    y, x = np.where(permitted & np.isfinite(disparity) &
                    (disparity + doffs > 0))
    disparity = disparity[y, x]
    target_x = np.rint(x - disparity).astype(np.int32)
    in_frame = (target_x >= 0) & (target_x < width)
    x, y, target_x, disparity = (array[in_frame] for array in (x, y, target_x, disparity))
    depth_mm = focal * baseline_mm / (disparity + doffs)
    target_linear = y * width + target_x
    # Sort target address first, then nearest surface first. The first entry
    # for each address is its visible z-buffer sample.
    ordering = np.lexsort((depth_mm, target_linear))
    sorted_addresses = target_linear[ordering]
    _, first = np.unique(sorted_addresses, return_index=True)
    source_ids = ordering[first]
    canvas = np.zeros((height * width,), dtype=np.uint8)
    coverage = np.zeros((height * width,), dtype=bool)
    address = target_linear[source_ids]
    canvas[address] = left[y[source_ids], x[source_ids]]
    coverage[address] = True
    return canvas.reshape(height, width), coverage.reshape(height, width)


def write_ply(path: Path, positions: np.ndarray, colors: np.ndarray,
              faces: np.ndarray | None) -> None:
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write("comment ETH3D CC BY-NC-SA 4.0; derived from delivery_area_1s laser-reference disparity\n")
        stream.write("comment local camera frame: X right, Y down, Z forward; coordinates in metres\n")
        stream.write(f"element vertex {len(positions)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if faces is not None:
            stream.write(f"element face {len(faces)}\n")
            stream.write("property list uchar int vertex_indices\n")
        stream.write("end_header\n")
        for xyz, gray in zip(positions, colors):
            stream.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} {int(gray)} {int(gray)} {int(gray)}\n")
        if faces is not None:
            for a, b, c in faces:
                stream.write(f"3 {a} {b} {c}\n")


def build_partial_mesh(left: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                       calib: dict, stride: int, rectangle: tuple[int, int, int, int],
                       depth_under_m: float, shift_toward_m: float) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    height, width = left.shape
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sampled_d = gt[yy, xx]
    valid = (np.isfinite(sampled_d) & (sampled_d + calib["doffs"] > 0) &
             (mask[yy, xx] == 255))
    Z_mm = np.zeros_like(sampled_d, dtype=np.float64)
    Z_mm[valid] = calib["K_left"][0, 0] * calib["baseline"] / (
        sampled_d[valid] + calib["doffs"])
    xyz = np.zeros((*sampled_d.shape, 3), dtype=np.float64)
    xyz[..., 0][valid] = (xx[valid] - calib["K_left"][0, 2]) / calib["K_left"][0, 0] * Z_mm[valid] / 1000
    xyz[..., 1][valid] = (yy[valid] - calib["K_left"][1, 2]) / calib["K_left"][1, 1] * Z_mm[valid] / 1000
    xyz[..., 2][valid] = Z_mm[valid] / 1000
    old_to_new = np.full(valid.shape, -1, dtype=np.int32)
    old_to_new[valid] = np.arange(valid.sum(), dtype=np.int32)
    verts = xyz[valid].astype(np.float32)
    gray = left[yy[valid], xx[valid]]
    faces = []
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            for cells in (((i, j), (i + 1, j), (i, j + 1)),
                          ((i + 1, j), (i + 1, j + 1), (i, j + 1))):
                if not all(valid[a, b] for a, b in cells):
                    continue
                depths = [Z_mm[a, b] for a, b in cells]
                if max(depths) / min(depths) > 1.08:
                    continue  # no artificial triangle across a large depth jump
                faces.append([old_to_new[a, b] for a, b in cells])
    edited = verts.copy()
    # Hand-picked image rectangle AND depth condition. This is a geometric
    # patch edit, not semantic segmentation of the entire crate.
    xmin, ymin, xmax, ymax = rectangle
    selected = (valid & (xx >= xmin) & (xx <= xmax) & (yy >= ymin) &
                (yy <= ymax) & (Z_mm < depth_under_m * 1000))
    edited[selected[valid], 2] -= shift_toward_m
    return verts, gray, np.asarray(faces, dtype=np.int32), edited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="delivery_area_1s")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--num-disparities", type=int, default=256)
    parser.add_argument("--edit-rectangle", nargs=4, type=int,
                        default=(170, 230, 600, 430), metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--edit-depth-under-m", type=float, default=3.0)
    parser.add_argument("--shift-toward-camera-m", type=float, default=0.15)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.scene != "delivery_area_1s" or args.stride < 1 or
            args.num_disparities <= 0 or args.num_disparities % 16):
        raise ValueError("this transparent mesh/edit example is fixed to inspected delivery_area_1s")
    xmin, ymin, xmax, ymax = args.edit_rectangle
    if (not (0 <= xmin < xmax < 711 and 0 <= ymin < ymax < 435) or
            args.edit_depth_under_m <= 0 or args.shift_toward_camera_m <= 0):
        raise ValueError("invalid edit rectangle, depth cap, or positive shift")
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(out)
    directory = BASE / "images" / args.scene
    reference = BASE / "ground_truth" / args.scene
    left = read_gray(directory / "im0.png")
    right = read_gray(directory / "im1.png")
    gt = read_pfm(reference / "disp0GT.pfm")
    mask = read_gray(reference / "mask0nocc.png")
    calib = read_calib(directory / "calib.txt")
    evaluable = (mask == 255) & np.isfinite(gt) & (gt + calib["doffs"] > 0)
    sgbm = estimate(left, right, "semi_global", args.num_disparities)
    true_warp, true_coverage = forward_warp(left, gt, evaluable,
                                            calib["K_left"][0, 0], calib["baseline"], calib["doffs"])
    estimated_warp, estimated_coverage = forward_warp(left, sgbm, sgbm > 0,
                                                      calib["K_left"][0, 0], calib["baseline"], calib["doffs"])
    common = true_coverage & estimated_coverage
    true_mse_common = float(np.mean((true_warp[common].astype(np.float32) - right[common]) ** 2) / (255 ** 2))
    estimated_mse_common = float(np.mean((estimated_warp[common].astype(np.float32) - right[common]) ** 2) / (255 ** 2))
    verts, grays, faces, edited = build_partial_mesh(
        left, gt, mask, calib, args.stride, tuple(args.edit_rectangle),
        args.edit_depth_under_m, args.shift_toward_camera_m)
    selected_count = int(np.count_nonzero(np.abs(edited[:, 2] - verts[:, 2]) > 1e-8))
    out.mkdir(parents=True)
    write_ply(out / "reference_partial_points.ply", verts, grays, None)
    write_ply(out / "reference_partial_mesh.ply", verts, grays, faces)
    write_ply(out / "edited_patch_partial_mesh.ply", edited, grays, faces)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    panels = [
        (left, "左图：像素输入"), (right, "右图：真实要到达的视角"),
        (np.where(true_coverage, true_warp, 0), "用外部激光参考深度前向投影"),
        (np.where(estimated_coverage, estimated_warp, 0), "仅两图估视差后再投影"),
        (true_coverage.astype(np.uint8) * 255, "激光参考投影的填充位置"),
        (estimated_coverage.astype(np.uint8) * 255, "自估视差投影的填充位置"),
    ]
    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("真实双目：深度可把左图搬到右相机，但遮挡与空洞仍存在", fontsize=16)
    fig.savefig(out / "view_warp_with_holes.png", dpi=140)
    plt.close(fig)
    # A close side view makes the actual vertex displacement visible. The
    # separate PLY files retain all 3D coordinates and triangle topology.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    axes[0].imshow(left, cmap="gray", vmin=0, vmax=255)
    region = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                           edgecolor="red", linewidth=2)
    axes[0].add_patch(region)
    axes[0].set_title(f"手选图像区域；再限制深度小于{args.edit_depth_under_m:g}米")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    selected_vertices = np.abs(edited[:, 2] - verts[:, 2]) > 1e-8
    nearby = (verts[:, 2] < 4) & (verts[:, 2] > 1.5)
    for ax, data, label in ((axes[1], verts, "原网格顶点：侧面 X—Z"),
                            (axes[2], edited, "实际改动的顶点红色显示")):
        ax.scatter(data[nearby & ~selected_vertices, 0],
                   data[nearby & ~selected_vertices, 2],
                   s=0.3, alpha=0.2, c="black")
        ax.scatter(data[nearby & selected_vertices, 0],
                   data[nearby & selected_vertices, 2],
                   s=0.4, alpha=0.3, c="red")
        ax.set_xlim(float(np.percentile(verts[nearby, 0], 2)),
                    float(np.percentile(verts[nearby, 0], 98)))
        ax.set_ylim(1.5, 4)
        ax.set_xlabel("X：画面向右的相机坐标 / 米")
        ax.set_ylabel("Z：向前的相机坐标 / 米")
        ax.set_title(label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"激光参考视差反投影成部分三角网格；选中顶点向相机移{args.shift_toward_camera_m:g}米", fontsize=15)
    fig.savefig(out / "partial_geometry_and_edit.png", dpi=140)
    plt.close(fig)
    receipt = {
        "identity": "geometry-guided warp and partial point/triangle assets from ETH3D external laser-reference disparity; no network learned the ground-truth depth",
        "source_extraction_manifest_sha256": sha256(BASE / "extraction_manifest.json"),
        "scene": args.scene, "stride": args.stride,
        "sgbm_num_disparities": args.num_disparities,
        "sgbm_range_selection_identity": ("original 256-range" if args.num_disparities == 256 else
                                          "post-hoc narrower range after GT metrics inspected"),
        "calibration": {"fx_pixels": float(calib["K_left"][0, 0]),
                        "fy_pixels": float(calib["K_left"][1, 1]),
                        "baseline_mm": calib["baseline"], "doffs_pixels": calib["doffs"]},
        "warp": {"reference_filled_pixels": int(true_coverage.sum()),
                 "estimated_filled_pixels": int(estimated_coverage.sum()),
                 "common_filled_pixels": int(common.sum()),
                 "reference_mse_on_common_filled_rgb01": true_mse_common,
                 "estimated_mse_on_common_filled_rgb01": estimated_mse_common,
                 "target_pixels": left.size,
                 "rule": "rounded forward splat to right u = left u - disparity; nearest depth wins collisions; holes left black; common-filled MSE only and separate coverage"},
        "geometry": {"vertices": len(verts), "faces": len(faces),
                     "hand_selected_edited_vertices": selected_count,
                     "edit": f"for visible {xmin}<=u<={xmax},{ymin}<=v<={ymax} and Z<{args.edit_depth_under_m:g}m, subtract {args.shift_toward_camera_m:g}m from camera Z; not semantic object segmentation, not a closed model"},
        "outputs_sha256": {path.name: sha256(path) for path in out.iterdir() if path.is_file()},
        "limits": "GT comes from registered laser scan, not from left/right algorithm; one view yields partial 2.5D geometry with holes and no unseen back surfaces; author hand edit may break topology at boundary, and is not a generated 3D object asset.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"warp": receipt["warp"], "geometry": receipt["geometry"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
