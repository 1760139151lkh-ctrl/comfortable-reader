"""下载并核对官方 Fashion-MNIST；保存本章固定划分和像素置换。"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data" / "fashion_mnist"
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
SPLIT_SEED = 20260924
PERMUTATION_SEED = 20260925


def digest(data: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, data).hexdigest()


def download_and_check(name: str, expected_md5: str) -> dict:
    path = DATA / name
    if not path.exists():
        request = urllib.request.Request(BASE + name, headers={"User-Agent": "AI-textbook-local-verification/1.0"})
        temporary = DATA / (name + ".part")
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(path)
    raw = path.read_bytes()
    actual_md5 = digest(raw, "md5")
    if actual_md5 != expected_md5:
        raise ValueError(f"{name}: 官方 MD5 不符，实际 {actual_md5}")
    return {
        "path": str(path.relative_to(WORK)).replace("\\", "/"),
        "url": BASE + name,
        "bytes": len(raw),
        "md5": actual_md5,
        "sha256": digest(raw, "sha256"),
    }


def load_images(path: Path, expected_count: int) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < 16:
        raise ValueError("图像 IDX 过短")
    magic, count, height, width = struct.unpack(">IIII", raw[:16])
    if (magic, count, height, width) != (2051, expected_count, 28, 28):
        raise ValueError(f"图像 IDX 标头不符：{(magic, count, height, width)}")
    if len(raw) != 16 + count * height * width:
        raise ValueError("图像 IDX 字节数与标头不符")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, height, width).copy()


def load_labels(path: Path, expected_count: int) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < 8:
        raise ValueError("标签 IDX 过短")
    magic, count = struct.unpack(">II", raw[:8])
    if (magic, count) != (2049, expected_count):
        raise ValueError(f"标签 IDX 标头不符：{(magic, count)}")
    if len(raw) != 8 + count:
        raise ValueError("标签 IDX 字节数与标头不符")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    if np.any(labels > 9):
        raise ValueError("标签不在 0..9")
    return labels


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    identities = {name: download_and_check(name, md5) for name, md5 in FILES.items()}
    train_x = load_images(DATA / "train-images-idx3-ubyte.gz", 60_000)
    train_y = load_labels(DATA / "train-labels-idx1-ubyte.gz", 60_000)
    test_x = load_images(DATA / "t10k-images-idx3-ubyte.gz", 10_000)
    test_y = load_labels(DATA / "t10k-labels-idx1-ubyte.gz", 10_000)
    if not np.array_equal(np.bincount(train_y, minlength=10), np.full(10, 6000)):
        raise ValueError("官方训练类别计数不是每类 6000")
    if not np.array_equal(np.bincount(test_y, minlength=10), np.full(10, 1000)):
        raise ValueError("官方测试类别计数不是每类 1000")

    rng = np.random.default_rng(SPLIT_SEED)
    train_indices, validation_indices = [], []
    for label in range(10):
        class_indices = rng.permutation(np.flatnonzero(train_y == label))
        validation_indices.extend(class_indices[:1000].tolist())
        train_indices.extend(class_indices[1000:].tolist())
    train_indices = np.asarray(sorted(train_indices), dtype=np.int32)
    validation_indices = np.asarray(sorted(validation_indices), dtype=np.int32)
    if len(train_indices) != 50_000 or len(validation_indices) != 10_000:
        raise AssertionError("训练／验证行数不符")
    if np.intersect1d(train_indices, validation_indices).size:
        raise AssertionError("训练与验证重叠")
    if np.union1d(train_indices, validation_indices).size != 60_000:
        raise AssertionError("有官方训练行遗失")
    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(784).astype(np.int32)
    if not np.array_equal(np.sort(permutation), np.arange(784)):
        raise AssertionError("像素映射不是双射")

    split_file = DATA / "chapter_split.npz"
    np.savez(split_file, train_indices=train_indices, validation_indices=validation_indices, permutation=permutation)
    metadata = {
        "official_readme": "https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md",
        "official_2017_paper": "https://arxiv.org/abs/1708.07747",
        "files": identities,
        "format": "IDX gzip; uint8 28x28 images, uint8 labels 0..9",
        "split_seed": SPLIT_SEED,
        "permutation_seed": PERMUTATION_SEED,
        "train_counts": np.bincount(train_y[train_indices], minlength=10).tolist(),
        "validation_counts": np.bincount(train_y[validation_indices], minlength=10).tolist(),
        "official_test_counts": np.bincount(test_y, minlength=10).tolist(),
        "split_file": str(split_file.relative_to(WORK)).replace("\\", "/"),
        "split_file_sha256": digest(split_file.read_bytes(), "sha256"),
        "first_train_image_pixel_sum": int(train_x[0].sum()),
        "first_test_image_pixel_sum": int(test_x[0].sum()),
        "note": "官方测试未拆分、未参与训练或验证选择；项目 README 声明 MIT 许可，图像权属与更广泛用途请见原项目。"
    }
    (DATA / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("train_counts", "validation_counts", "official_test_counts", "split_file_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
