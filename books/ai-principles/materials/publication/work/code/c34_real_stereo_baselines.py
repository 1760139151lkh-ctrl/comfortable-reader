"""Run fixed, non-learned two-view stereo on all 27 real ETH3D training pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

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


def estimate(left: np.ndarray, right: np.ndarray, name: str,
             num_disparities: int = 256) -> np.ndarray:
    if name == "block_match":
        model = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=15)
    elif name == "semi_global":
        model = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities, blockSize=5,
            P1=8 * 5 * 5, P2=32 * 5 * 5,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=50, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    else:
        raise ValueError(name)
    return model.compute(left, right).astype(np.float32) / 16.0


def measure(pred: np.ndarray, truth: np.ndarray,
            mask: np.ndarray, calib: dict) -> tuple[dict, np.ndarray]:
    evaluable = np.isfinite(truth) & (truth + calib["doffs"] > 0) & (mask == 255)
    predicted = np.isfinite(pred) & (pred + calib["doffs"] > 0)
    both = evaluable & predicted
    if not evaluable.any():
        raise RuntimeError("no usable reference disparity")
    abs_error = np.full(truth.shape, np.nan, dtype=np.float32)
    abs_error[both] = np.abs(pred[both] - truth[both])
    # Invalid estimates count as failures on the true double-visible pixels.
    bad1 = evaluable & (~predicted | (np.abs(pred - truth) > 1.0))
    bad3 = evaluable & (~predicted | (np.abs(pred - truth) > 3.0))
    true_count = int(evaluable.sum())
    accepted = int(both.sum())
    return {
        "evaluable_gt_pixels": true_count,
        "predicted_valid_among_evaluable": accepted,
        "coverage": accepted / true_count,
        "bad_gt_1px_including_missing": int(bad1.sum()),
        "bad_gt_3px_including_missing": int(bad3.sum()),
        "bad1_rate_including_missing": float(bad1.sum() / true_count),
        "bad3_rate_including_missing": float(bad3.sum() / true_count),
        "median_abs_px_on_predicted_valid": float(np.median(abs_error[both])) if accepted else None,
        "mean_abs_px_on_predicted_valid": float(np.mean(abs_error[both])) if accepted else None,
        "ground_truth_median_disparity_px": float(np.median(truth[evaluable])),
        "ground_truth_median_depth_in_baseline_units": float(np.median(
            calib["K_left"][0, 0] * calib["baseline"] /
            (truth[evaluable] + calib["doffs"]))),
    }, abs_error


def plot_example(out: Path, scene: str, left: np.ndarray, right: np.ndarray,
                 truth: np.ndarray, mask: np.ndarray,
                 predictions: dict[str, np.ndarray],
                 errors: dict[str, np.ndarray]) -> str:
    available = np.isfinite(truth) & (mask == 255) & (truth > 0)
    upper = float(np.percentile(truth[available], 98))
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    panels = [
        (left, "左图：算法输入", "gray", 0, 255),
        (right, "右图：算法输入", "gray", 0, 255),
        (np.where(available, truth, np.nan), "激光参考视差：只供评价", "viridis", 0, upper),
        (np.where(predictions["block_match"] > 0, predictions["block_match"], np.nan),
         "局部方块匹配", "viridis", 0, upper),
        (np.where(predictions["semi_global"] > 0, predictions["semi_global"], np.nan),
         "半全局路径平滑匹配", "viridis", 0, upper),
        (errors["semi_global"], "半全局绝对视差误差：有效预测处", "magma", 0, 8),
    ]
    for ax, (array, title, cmap, lo, hi) in zip(axes.flat, panels):
        im = ax.imshow(array, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if cmap != "gray":
            figure.colorbar(im, ax=ax, fraction=0.045, label="像素")
    figure.suptitle(f"ETH3D {scene}：同一真实场景的两视图与深度证据", fontsize=16)
    target = out / "stereo_real_example.png"
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return sha256(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--example", default="delivery_area_1s")
    parser.add_argument("--num-disparities", type=int, default=256)
    args = parser.parse_args()
    if args.num_disparities <= 0 or args.num_disparities % 16:
        raise ValueError("OpenCV numDisparities must be a positive multiple of 16")
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(out)
    scenes = sorted(p.name for p in (BASE / "images").iterdir() if p.is_dir())
    if len(scenes) != 27 or args.example not in scenes:
        raise RuntimeError("must use all 27 official training pairs")
    out.mkdir(parents=True)
    methods = ("block_match", "semi_global")
    report = {
        "identity": ("first fixed-parameter author evaluation" if args.num_disparities == 256 else
                     "post-hoc disparity search-range revision after reading first 256-range GT metrics") +
                    " of non-learned OpenCV stereo on all 27 ETH3D two-view TRAIN scene pairs; not ETH3D official hidden test or new NeRF model",
        "source_extraction_manifest_sha256": sha256(BASE / "extraction_manifest.json"),
        "opencv_version": cv2.__version__,
        "parameters": {"block_match": f"StereoBM numDisparities={args.num_disparities} blockSize=15",
                       "semi_global": f"StereoSGBM numDisparities={args.num_disparities} blockSize=5 P1=200 P2=800 disp12MaxDiff=1 uniquenessRatio=10 speckleWindow=50 speckleRange=2 MODE_SGBM_3WAY"},
        "evaluation": "only laser-reference finite positive disparity and mask0nocc=255; missing/invalid algorithm disparity counts bad in Bad1/Bad3, coverage reported separately; no GT input to algorithms or parameter selection",
        "scenes": {},
    }
    sums = {name: {"gt": 0, "covered": 0, "bad1": 0, "bad3": 0} for name in methods}
    for scene in scenes:
        directory = BASE / "images" / scene
        reference = BASE / "ground_truth" / scene
        left = read_gray(directory / "im0.png")
        right = read_gray(directory / "im1.png")
        gt = read_pfm(reference / "disp0GT.pfm")
        mask = read_gray(reference / "mask0nocc.png")
        calib = read_calib(directory / "calib.txt")
        if left.shape != right.shape or left.shape != gt.shape or left.shape != mask.shape:
            raise RuntimeError(f"shape mismatch in {scene}")
        row = {"camera": {"fx_pixels": float(calib["K_left"][0, 0]),
                           "baseline_as_calib": calib["baseline"],
                           "doffs_pixels": calib["doffs"]}, "methods": {}}
        predictions, errors = {}, {}
        for method in methods:
            pred = estimate(left, right, method, args.num_disparities)
            metrics, error = measure(pred, gt, mask, calib)
            row["methods"][method] = metrics
            predictions[method], errors[method] = pred, error
            sums[method]["gt"] += metrics["evaluable_gt_pixels"]
            sums[method]["covered"] += metrics["predicted_valid_among_evaluable"]
            sums[method]["bad1"] += metrics["bad_gt_1px_including_missing"]
            sums[method]["bad3"] += metrics["bad_gt_3px_including_missing"]
        report["scenes"][scene] = row
        print(scene, {m: round(row["methods"][m]["bad1_rate_including_missing"], 3)
                      for m in methods}, flush=True)
        if scene == args.example:
            report["example_figure_sha256"] = plot_example(out, scene, left, right,
                                                            gt, mask, predictions, errors)
            report["example_scene"] = scene
    report["aggregate"] = {
        method: {"gt_pixels": s["gt"], "coverage": s["covered"] / s["gt"],
                 "bad1_rate_including_missing": s["bad1"] / s["gt"],
                 "bad3_rate_including_missing": s["bad3"] / s["gt"],
                 "mean_of_scene_bad1_rates": float(np.mean([
                     report["scenes"][scene]["methods"][method]["bad1_rate_including_missing"]
                     for scene in scenes]))}
        for method, s in sums.items()}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print("aggregate", report["aggregate"], flush=True)


if __name__ == "__main__":
    main()
