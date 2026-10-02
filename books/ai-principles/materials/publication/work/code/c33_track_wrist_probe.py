"""Track one real training-video wrist from a human seed with LK-family flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/penn_action_original"
VIDEO = "0002"
START = 30  # zero-based frame index, official train split
END = 48
JOINT = 6  # right wrist


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gray(index: int) -> np.ndarray:
    path = DATA / "extracted/Penn_Action/frames" / VIDEO / f"{index + 1:06d}.jpg"
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot decode frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    figure = args.figure if args.figure.is_absolute() else ROOT / args.figure
    if out.exists() or figure.exists():
        raise FileExistsError("preserve existing tracker probe artifacts")
    inventory = json.loads((DATA / "inventory.json").read_text(encoding="utf-8"))
    row = next(item for item in inventory["rows"] if item["id"] == VIDEO)
    if row["official_split"] != "train":
        raise RuntimeError("tracking probe must use a training video")
    label_path = DATA / "extracted/Penn_Action/labels" / (VIDEO + ".mat")
    annotation = loadmat(label_path, squeeze_me=True)
    x = np.asarray(annotation["x"], dtype=float)
    y = np.asarray(annotation["y"], dtype=float)
    visible = np.asarray(annotation["visibility"]).astype(bool)
    if not visible[START:END + 1, JOINT].all():
        raise RuntimeError("right wrist ceases to be visible in chosen interval")
    previous_image = gray(START)
    point = np.array([[[x[START, JOINT], y[START, JOINT]]]], dtype=np.float32)
    positions = [(START, float(point[0, 0, 0]), float(point[0, 0, 1]))]
    statuses = []
    for index in range(START + 1, END + 1):
        current_image = gray(index)
        next_point, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_image, current_image, point, None,
            winSize=(31, 31), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_point is None or status is None or not bool(status[0, 0]):
            statuses.append(False)
            break
        statuses.append(True)
        point = next_point
        positions.append((index, float(point[0, 0, 0]), float(point[0, 0, 1])))
        previous_image = current_image
    indices = np.array([item[0] for item in positions], dtype=int)
    tracked = np.array([[item[1], item[2]] for item in positions])
    truth = np.stack([x[indices, JOINT], y[indices, JOINT]], axis=1)
    initial = truth[0]
    track_errors = np.linalg.norm(tracked - truth, axis=1)
    fixed_errors = np.linalg.norm(initial[None] - truth, axis=1)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    axes[0].plot(indices + 1, truth[:, 0], "o-", label="人工右手腕 x")
    axes[0].plot(indices + 1, tracked[:, 0], "o-", label="单初值的LK逐帧跟踪 x")
    axes[0].axhline(initial[0], color="#888888", linestyle="--", label="始终报第一帧 x")
    axes[0].set(xlabel="视频帧序号", ylabel="图像中的横坐标（像素）",
                title="同一真实棒球投球序列：跟踪与已知人工点")
    axes[0].legend()
    axes[1].plot(indices + 1, track_errors, "o-", label="LK对人工标注的像素误差")
    axes[1].plot(indices + 1, fixed_errors, "o-", label="固定起点对人工标注的像素误差")
    axes[1].set(xlabel="视频帧序号", ylabel="二维像素距离",
                title="两条方法都只从第31帧给一个初始位置")
    axes[1].legend()
    figure.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    report = {
        "source_archive_sha256": inventory["source_archive_sha256"],
        "label_sha256": sha256(label_path),
        "video_id": VIDEO, "official_split": "train",
        "action": str(annotation["action"]),
        "joint": "right wrist, zero-based column 6",
        "start_frame_zero_based": START, "requested_end_frame_zero_based": END,
        "last_successful_frame_zero_based": int(indices[-1]),
        "step_statuses": statuses,
        "opencv_version": cv2.__version__,
        "tracker": "OpenCV pyramidal Lucas-Kanade family, winSize31x31,maxLevel3, initialized ONCE from human mark at start; no supervised training",
        "human_label_xy": truth.tolist(),
        "tracked_xy": tracked.tolist(),
        "tracked_error_pixels_by_frame": track_errors.tolist(),
        "fixed_start_error_pixels_by_frame": fixed_errors.tolist(),
        "tracked_mean_error_after_initial": float(track_errors[1:].mean()) if len(indices) > 1 else None,
        "fixed_mean_error_after_initial": float(fixed_errors[1:].mean()) if len(indices) > 1 else None,
        "tracked_final_error_pixels": float(track_errors[-1]),
        "fixed_final_error_pixels": float(fixed_errors[-1]),
        "limits": "one author-selected training clip; human initial wrist coordinate supplied; brightness/texture may drift or fail, image motion not 3-D physical velocity",
        "figure_sha256": sha256(figure),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"last_frame": int(indices[-1]),
                      "lk_mean_error": report["tracked_mean_error_after_initial"],
                      "fixed_mean_error": report["fixed_mean_error_after_initial"],
                      "lk_final_error": report["tracked_final_error_pixels"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
