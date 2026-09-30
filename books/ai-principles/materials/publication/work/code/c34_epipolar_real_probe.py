"""Evaluate the epipolar equation on actual ETH3D/COLMAP matched image points."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from c34_multiview_reprojection import observed_by_image


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/eth3d_pipes/views_518"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def image_xy(view: dict, rows: list[tuple[float, float, int]]) -> dict[int, np.ndarray]:
    out = {}
    for u, v, point_id in rows:
        if point_id < 0:
            continue
        out.setdefault(point_id, np.array([u, v, 1.0]))
    return out


def pixel_line_distance(F: np.ndarray, left: np.ndarray,
                        right: np.ndarray) -> np.ndarray:
    lines = left @ F.T
    numer = np.abs(np.sum(lines * right, axis=1))
    denom = np.linalg.norm(lines[:, :2], axis=1)
    return numer / np.maximum(denom, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(out)
    source = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in source["views"]}
    a = by_name["DSC_0634.JPG"]
    b = by_name["DSC_0635.JPG"]
    observed = observed_by_image()
    left = image_xy(a, observed[a["image_id"]])
    right = image_xy(b, observed[b["image_id"]])
    shared = sorted(set(left) & set(right))
    if len(shared) < 20:
        raise RuntimeError("too few real matched points")
    p1 = np.stack([left[key] for key in shared])
    p2 = np.stack([right[key] for key in shared])
    cam = source["original_camera"]
    K = np.array([[cam["fx"], 0.0, cam["cx"]],
                  [0.0, cam["fy"], cam["cy"]],
                  [0.0, 0.0, 1.0]])
    Ra, Rb = np.asarray(a["rotation_world_to_camera"]), np.asarray(b["rotation_world_to_camera"])
    ta, tb = np.asarray(a["translation_world_to_camera"]), np.asarray(b["translation_world_to_camera"])
    R = Rb @ Ra.T
    t = tb - R @ ta
    E = skew(t) @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    correct = pixel_line_distance(F, p1, p2)
    rng = np.random.default_rng(20260924)
    shuffled = p2[rng.permutation(len(p2))]
    wrong = pixel_line_distance(F, p1, shuffled)
    index = int(np.argsort(correct)[len(correct) // 2])
    _, small_left, small_right = (None, DATA / "images" / Path(a["name"]).with_suffix(".png"),
                                  DATA / "images" / Path(b["name"]).with_suffix(".png"))
    def read_rgb(path: Path) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    left_image, right_image = read_rgb(small_left), read_rgb(small_right)
    scale_x = left_image.shape[1] / cam["width"]
    scale_y = left_image.shape[0] / cam["height"]
    out.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].imshow(left_image)
    axes[0].scatter([p1[index, 0] * scale_x], [p1[index, 1] * scale_y],
                    s=75, color="red", marker="+")
    axes[0].set_title("左图：一个真实匹配点")
    axes[1].imshow(right_image)
    line = F @ p1[index]
    x = np.array([0.0, cam["width"]])
    y = -(line[0] * x + line[2]) / line[1]
    axes[1].plot(x * scale_x, y * scale_y, color="yellow", linewidth=2,
                 label="由相机几何限定的极线")
    axes[1].scatter([p2[index, 0] * scale_x], [p2[index, 1] * scale_y],
                    s=75, color="cyan", marker="o", facecolors="none",
                    label="真实同点观测")
    axes[1].scatter([shuffled[index, 0] * scale_x], [shuffled[index, 1] * scale_y],
                    s=75, color="red", marker="x", label="故意错配")
    axes[1].set_title("右图：线缩小搜索范围，但不能指定线上的唯一点")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlim(0, left_image.shape[1])
        ax.set_ylim(left_image.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("ETH3D真实照片与上游SfM对应：两相机共面约束", fontsize=15)
    fig.savefig(out / "real_epipolar_line.png", dpi=140)
    plt.close(fig)
    receipt = {
        "identity": "actual matched 2D SfM observations and upstream supplied calibrated camera poses; internal geometric check, not independent 3D GT",
        "source_manifest_sha256": sha256(DATA / "manifest.json"),
        "pair": [a["name"], b["name"]],
        "shared_3d_track_ids": len(shared),
        "relative_rotation_world_to_camera": R.tolist(),
        "relative_translation": t.tolist(),
        "essential_matrix": E.tolist(), "fundamental_matrix": F.tolist(),
        "correct_pixel_distance_to_right_epipolar_line": {
            "median": float(np.median(correct)), "p95": float(np.percentile(correct, 95))},
        "permuted_wrong_pair_distance_pixels": {
            "median": float(np.median(wrong)), "p95": float(np.percentile(wrong, 95))},
        "selected_point_id": shared[index],
        "selected_correct_distance_pixels": float(correct[index]),
        "selected_wrong_distance_pixels": float(wrong[index]),
        "figure_sha256": sha256(out / "real_epipolar_line.png"),
        "limits": "incorrect match can still lie on epipolar line; upstream SfM estimated poses/tracks jointly from photos, so line residual only checks internal consistency, not independent 3D truth",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matched": len(shared), "correct_median_px": receipt["correct_pixel_distance_to_right_epipolar_line"]["median"],
                      "wrong_median_px": receipt["permuted_wrong_pair_distance_pixels"]["median"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
