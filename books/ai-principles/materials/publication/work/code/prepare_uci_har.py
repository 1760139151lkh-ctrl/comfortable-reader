"""核 UCI HAR 官方双层 ZIP，保存六路时间信号及受试者级划分。"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data" / "uci_har"
OUTER_ZIP = DATA / "UCI_HAR_Dataset.zip"
URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
EXPECTED_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
BASE = "UCI HAR Dataset/"
CHANNELS = ("body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z")
SPLIT_SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_outer_zip() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not OUTER_ZIP.exists():
        temporary = DATA / "UCI_HAR_Dataset.zip.part"
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(OUTER_ZIP)
    if digest(OUTER_ZIP) != EXPECTED_SHA256:
        raise ValueError("UCI 官方外层 ZIP 字节身份不符")


def read_table(zipped: zipfile.ZipFile, member: str, dtype: type) -> np.ndarray:
    return np.loadtxt(io.BytesIO(zipped.read(BASE + member)), dtype=dtype)


def read_part(zipped: zipfile.ZipFile, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = []
    for channel in CHANNELS:
        member = f"{part}/Inertial Signals/{channel}_{part}.txt"
        column = read_table(zipped, member, np.float32)
        if column.ndim != 2 or column.shape[1] != 128:
            raise ValueError(f"{member}: 不是 N×128")
        columns.append(column)
    signal = np.stack(columns, axis=-1)
    labels = read_table(zipped, f"{part}/y_{part}.txt", np.int64)
    subjects = read_table(zipped, f"{part}/subject_{part}.txt", np.int64)
    if len(signal) != len(labels) or len(signal) != len(subjects):
        raise ValueError(f"{part}: 时间窗、标签、受试者行数不同")
    if not np.isfinite(signal).all() or not np.isin(labels, np.arange(1, 7)).all():
        raise ValueError(f"{part}: 数值或标签范围异常")
    return signal, labels.astype(np.int8) - 1, subjects.astype(np.int8)


def main() -> None:
    get_outer_zip()
    with zipfile.ZipFile(OUTER_ZIP) as outer:
        if set(outer.namelist()) != {"UCI HAR Dataset.names", "UCI HAR Dataset.zip"}:
            raise ValueError("UCI 外层成员与下载时核对的不一致")
        inner_bytes = outer.read("UCI HAR Dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        train_x, train_y, train_subject = read_part(inner, "train")
        test_x, test_y, test_subject = read_part(inner, "test")
        activity_names = inner.read(BASE + "activity_labels.txt").decode("utf-8", errors="replace").strip().splitlines()
        readme = inner.read(BASE + "README.txt").decode("utf-8", errors="replace")
    if train_x.shape != (7352, 128, 6) or test_x.shape != (2947, 128, 6):
        raise ValueError(f"官方窗口数与维度不符：{train_x.shape}、{test_x.shape}")
    if np.intersect1d(np.unique(train_subject), np.unique(test_subject)).size:
        raise ValueError("官方训练／测试受试者重叠")
    if len(activity_names) != 6:
        raise ValueError("六类活动名称缺失")

    all_training_subjects = np.unique(train_subject)
    validation_subjects = np.sort(
        np.random.default_rng(SPLIT_SEED).choice(all_training_subjects, size=4, replace=False)
    )
    validation_mask = np.isin(train_subject, validation_subjects)
    train_indices = np.flatnonzero(~validation_mask).astype(np.int32)
    validation_indices = np.flatnonzero(validation_mask).astype(np.int32)
    if not np.all(np.bincount(train_y[train_indices], minlength=6) > 0):
        raise ValueError("训练部分有类别缺失")
    if not np.all(np.bincount(train_y[validation_indices], minlength=6) > 0):
        raise ValueError("验证部分有类别缺失")

    mean = train_x[train_indices].mean(axis=(0, 1), dtype=np.float64)
    std = train_x[train_indices].std(axis=(0, 1), dtype=np.float64)
    if np.any(std <= 0):
        raise ValueError("训练通道有零方差")
    prepared_file = DATA / "har_sequences.npz"
    np.savez_compressed(
        prepared_file,
        train_x=train_x,
        train_y=train_y,
        train_subject=train_subject,
        test_x=test_x,
        test_y=test_y,
        test_subject=test_subject,
        train_indices=train_indices,
        validation_indices=validation_indices,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    metadata = {
        "source_page": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        "source_url": URL,
        "citation": "Reyes-Ortiz, Anguita, Ghio, Oneto & Parra (2013), UCI HAR Using Smartphones, doi:10.24432/C54S4K",
        "license": "CC BY 4.0 according to UCI dataset page",
        "outer_zip_sha256": digest(OUTER_ZIP),
        "inner_zip_sha256": hashlib.sha256(inner_bytes).hexdigest(),
        "inner_readme_mentions_50_percent_overlap": "50% of overlap" in readme or "50% overlap" in readme,
        "channels": list(CHANNELS),
        "shape_train_official": list(train_x.shape),
        "shape_test_official": list(test_x.shape),
        "train_subject_ids": np.unique(train_subject[train_indices]).tolist(),
        "validation_subject_ids": validation_subjects.tolist(),
        "official_test_subject_ids": np.unique(test_subject).tolist(),
        "split_seed": SPLIT_SEED,
        "training_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "official_test_rows": len(test_y),
        "train_label_counts": np.bincount(train_y[train_indices], minlength=6).tolist(),
        "validation_label_counts": np.bincount(train_y[validation_indices], minlength=6).tolist(),
        "official_test_label_counts": np.bincount(test_y, minlength=6).tolist(),
        "activity_names_original": activity_names,
        "channel_mean_from_training_subjects": mean.tolist(),
        "channel_std_from_training_subjects": std.tolist(),
        "prepared_file": str(prepared_file.relative_to(WORK)).replace("\\", "/"),
        "prepared_file_sha256": digest(prepared_file),
        "note": "官方已滤噪并按 128 时刻和 50% 重叠切窗；真实测试按受试者保留，验证只从官方训练受试者划出。"
    }
    (DATA / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("training_rows", "validation_rows", "official_test_rows", "validation_subject_ids", "train_label_counts", "validation_label_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
