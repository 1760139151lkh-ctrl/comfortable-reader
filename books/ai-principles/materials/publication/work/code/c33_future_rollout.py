"""Roll a one-step Penn predictor forward four times on a real validation clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from c33_future_frame import FutureNet
from prepare_c33_real_motion import letterbox


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/penn_action_original"
MOTION = DATA / "motion_gap5_v1"
DISPLAY_FPS = 4  # only a preview clock; source real FPS is unknown


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame(video_id: str, index: int) -> np.ndarray:
    path = DATA / "extracted/Penn_Action/frames" / video_id / f"{index + 1:06d}.jpg"
    decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError(f"unreadable real frame: {path}")
    return letterbox(decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing rollout: {out}")
    manifest_path = MOTION / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = json.loads((run / "train.json").read_text(encoding="utf-8"))
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (training["data_manifest_sha256"] != sha256(manifest_path) or
            training["best_checkpoint_sha256"] != sha256(checkpoint_path) or
            checkpoint["data_manifest_sha256"] != sha256(manifest_path)):
        raise RuntimeError("future model/data mismatch")
    selected = manifest["metadata"]["validation"][0]
    video_id = selected["video_id"]
    center = selected["center_frame_index_zero_based"]
    if video_id != "0005" or center != 23:
        raise RuntimeError("fixed validation rollout selection changed")
    inventory = json.loads((DATA / "inventory.json").read_text(encoding="utf-8"))
    row = next(item for item in inventory["rows"] if item["id"] == video_id)
    if row["official_split"] != "train" or row["nframes"] <= center + 20:
        raise RuntimeError("rollout source video insufficient")
    model = FutureNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.set_num_threads(4)
    past = torch.from_numpy(frame(video_id, center - 5).transpose(2, 0, 1)[None].astype(np.float32) / 255)
    current = torch.from_numpy(frame(video_id, center).transpose(2, 0, 1)[None].astype(np.float32) / 255)
    initial = current.clone()
    actual = []
    generated = []
    records = []
    with torch.no_grad():
        for step in range(1, 5):
            next_index = center + 5 * step
            prediction = model(past, current)
            true = torch.from_numpy(frame(video_id, next_index).transpose(2, 0, 1)[None].astype(np.float32) / 255)
            prediction_np = prediction[0].permute(1, 2, 0).numpy()
            true_np = true[0].permute(1, 2, 0).numpy()
            generated.append(prediction_np)
            actual.append(true_np)
            records.append({"step": step, "real_frame_index_zero_based": next_index,
                            "model_mse_per_rgb_coordinate": float(np.mean((prediction_np - true_np) ** 2)),
                            "copy_initial_current_mse_per_rgb_coordinate": float(
                                np.mean((initial[0].permute(1, 2, 0).numpy() - true_np) ** 2))})
            past, current = current, prediction
    out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 5, figsize=(15, 5), layout="constrained")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    current_image = initial[0].permute(1, 2, 0).numpy()
    for axis in (axes[0, 0], axes[1, 0]):
        axis.imshow(current_image)
        axis.set_title(f"已给当前第{center + 1}帧")
        axis.axis("off")
    for column, (truth, guess, record) in enumerate(zip(actual, generated, records), 1):
        axes[0, column].imshow(truth)
        axes[0, column].set_title(f"真实第{record['real_frame_index_zero_based'] + 1}帧")
        axes[1, column].imshow(np.clip(guess, 0, 1))
        axes[1, column].set_title(f"模型第{record['step']}步\nMSE {record['model_mse_per_rgb_coordinate']:.4f}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.suptitle("同一真实验证视频：一步预测回送为下一步输入；没有接入真实未来", fontsize=14)
    figure = out / "four_step_real_vs_generated.png"
    fig.savefig(figure, dpi=155)
    plt.close(fig)
    frames_dir = out / "generated_preview_frames"
    frames_dir.mkdir()
    for index, picture in enumerate([current_image, *generated], 1):
        Image.fromarray((np.clip(picture, 0, 1) * 255).astype(np.uint8)).save(
            frames_dir / f"{index:06d}.png")
    mp4 = out / "generated_preview_4fps.mp4"
    done = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                           "-framerate", str(DISPLAY_FPS), "-start_number", "1",
                           "-i", str(frames_dir / "%06d.png"),
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg rollout failed: {done.stderr[-800:]}")
    report = {
        "identity": "four autoregressive applications of one-step model on a real author-validation video, starting from two actual frames; not text-to-video or world simulation",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "data_manifest_sha256": sha256(manifest_path),
        "train_report_sha256": sha256(run / "train.json"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "validation_video_id": video_id,
        "original_action": row["action"],
        "original_official_split": "train",
        "author_partition": "validation",
        "initial_past_and_current_frame_indices_zero_based": [center - 5, center],
        "predicted_targets_gap_in_frame_indices": 5,
        "records": records,
        "chosen_preview_fps_not_source_camera_fps": DISPLAY_FPS,
        "source_real_fps": "UNKNOWN",
        "figure_sha256": sha256(figure),
        "mp4_sha256": sha256(mp4),
        "limits": "same validation source used to select one-step model; rolling out beyond trained single horizon accumulates error; MSE/copy are pixel-level and no human video preference was collected",
    }
    (out / "rollout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"video": video_id, "mse_by_step": [r["model_mse_per_rgb_coordinate"] for r in records]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
