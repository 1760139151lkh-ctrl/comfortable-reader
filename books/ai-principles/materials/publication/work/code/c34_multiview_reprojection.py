"""Check supplied ETH3D/COLMAP camera poses against their observed sparse 3D points."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/eth3d_pipes/views_518"
CALIB = ROOT / "work/data/eth3d_pipes/extracted/pipes/dslr_calibration_undistorted"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def points() -> dict[int, np.ndarray]:
    result = {}
    for line in (CALIB / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        columns = line.split()
        result[int(columns[0])] = np.array([float(x) for x in columns[1:4]])
    if len(result) != 2473:
        raise RuntimeError("unexpected SfM point count")
    return result


def observed_by_image() -> dict[int, list[tuple[float, float, int]]]:
    lines = [line.strip() for line in (CALIB / "images.txt").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    if len(lines) != 28:
        raise RuntimeError("unexpected COLMAP image row pairs")
    out = {}
    for i in range(0, len(lines), 2):
        pose = lines[i].split()
        obs = lines[i + 1].split()
        if len(obs) % 3:
            raise RuntimeError("2D observation triples malformed")
        out[int(pose[0])] = [(float(obs[j]), float(obs[j + 1]), int(obs[j + 2]))
                             for j in range(0, len(obs), 3)]
    return out


def project(xyz_world: np.ndarray, view: dict,
            intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(view["rotation_world_to_camera"])
    t = np.asarray(view["translation_world_to_camera"])
    local = xyz_world @ R.T + t
    uv = np.empty((len(local), 2))
    uv[:, 0] = intrinsics["fx"] * local[:, 0] / local[:, 2] + intrinsics["cx"]
    uv[:, 1] = intrinsics["fy"] * local[:, 1] / local[:, 2] + intrinsics["cy"]
    return uv, local[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(out)
    source = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    vertices = points()
    observations = observed_by_image()
    K = source["original_camera"]
    residuals = []
    per_view = {}
    seen_point = defaultdict(list)
    for view in source["views"]:
        rows = [(u, v, pid) for u, v, pid in observations[view["image_id"]] if pid in vertices]
        X = np.stack([vertices[pid] for _, _, pid in rows])
        measured = np.array([[u, v] for u, v, _ in rows])
        predicted, depth = project(X, view, K)
        if np.any(depth <= 0):
            raise RuntimeError("registered SfM point behind own observing camera")
        errors = np.linalg.norm(predicted - measured, axis=1)
        residuals.extend(errors)
        per_view[view["name"]] = {"observed_triangulated_points": len(rows),
                                  "median_reprojection_error_pixels": float(np.median(errors)),
                                  "p95_error_pixels": float(np.percentile(errors, 95))}
        for (u, v, pid), predicted_uv, error in zip(rows, predicted, errors):
            seen_point[pid].append({"view_name": view["name"], "actual_uv": [u, v],
                                    "predicted_uv": predicted_uv.tolist(),
                                    "residual_pixels": float(error)})
    # Original observation records can repeat a 3D ID within one image. Keep
    # one lowest-residual item per actual camera so the figure has six views.
    for pid, rows in list(seen_point.items()):
        unique = {}
        for row in rows:
            name = row["view_name"]
            if name not in unique or row["residual_pixels"] < unique[name]["residual_pixels"]:
                unique[name] = row
        seen_point[pid] = list(unique.values())
    # Choose a point with many distinct camera views and low error.
    candidate = min(((-len(rows), np.median([r["residual_pixels"] for r in rows]), pid)
                     for pid, rows in seen_point.items() if len(rows) >= 6))[2]
    selected = seen_point[candidate][:6]
    views_by_name = {v["name"]: v for v in source["views"]}
    width, height = source["small_intrinsics"]["width"], source["small_intrinsics"]["height"]
    sx, sy = width / K["width"], height / K["height"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, row in zip(axes.flat, selected):
        view = views_by_name[row["view_name"]]
        bgr = cv2.imdecode(np.fromfile(ROOT / view["small_path"], dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        u, v = row["actual_uv"]
        pred_u, pred_v = row["predicted_uv"]
        ax.scatter([u * sx], [v * sy], marker="o", facecolors="none",
                   edgecolors="cyan", s=75, linewidths=1.5)
        ax.scatter([pred_u * sx], [pred_v * sy], marker="+", color="red", s=75)
        ax.set_xlim(max(0, u * sx - 75), min(width, u * sx + 75))
        ax.set_ylim(min(height, v * sy + 60), max(0, v * sy - 60))
        ax.set_title(f"{row['view_name']}：误差{row['residual_pixels']:.2f}原像素")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("同一已匹配三维点投到六张真实图：青圈原观测，红十字按相机式重投影", fontsize=15)
    out.mkdir(parents=True)
    fig.savefig(out / "one_point_six_views.png", dpi=140)
    plt.close(fig)
    errors = np.asarray(residuals)
    receipt = {
        "identity": "internal reprojection consistency of supplied ETH3D/COLMAP SfM points and observed 2D tracks; NOT independent laser-scan 3D accuracy",
        "data_manifest_sha256": sha256(DATA / "manifest.json"),
        "points3d_sha256": sha256(CALIB / "points3D.txt"),
        "images_txt_sha256": sha256(CALIB / "images.txt"),
        "matched_observations": len(errors),
        "median_reprojection_error_original_pixels": float(np.median(errors)),
        "p95_reprojection_error_original_pixels": float(np.percentile(errors, 95)),
        "per_view": per_view,
        "illustrated_point_id": candidate,
        "illustrated_point_world_coordinates": vertices[candidate].tolist(),
        "illustrated_views": selected,
        "figure_sha256": sha256(out / "one_point_six_views.png"),
        "limits": "SfM cameras and 3D points were jointly estimated upstream; low reprojection error does not independently prove true geometry or generalization to new views",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"matched": len(errors), "median_px": receipt["median_reprojection_error_original_pixels"],
                      "p95_px": receipt["p95_reprojection_error_original_pixels"],
                      "illustrated_id": candidate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
