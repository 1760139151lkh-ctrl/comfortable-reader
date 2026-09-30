"""核对 UCI Iris 原包，按 C08 协议固定行级切分。"""

import csv
import hashlib
import io
import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


WORK = Path(__file__).resolve().parents[1]
ARCHIVE = WORK / "data/uci_iris_original.zip"
RAW_DIR = WORK / "data/uci_iris"
RAW_FILE = RAW_DIR / "iris.data"
SPLIT_FILE = WORK / "data/uci_iris_split.json"
URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
ARCHIVE_SHA256 = "d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9"
RAW_SHA256 = "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"
SEED = 20260923
CLASS_TO_LABEL = {
    "Iris-setosa": None,
    "Iris-versicolor": -1,
    "Iris-virginica": 1,
}
QUOTAS = {"train": 30, "validation": 10, "test": 10}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_raw():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        ARCHIVE.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
    archive_bytes = ARCHIVE.read_bytes()
    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("UCI Iris ZIP 的字节身份发生变化")
    with ZipFile(ARCHIVE) as source:
        if set(source.namelist()) != {"Index", "bezdekIris.data", "iris.data", "iris.names"}:
            raise ValueError("ZIP 成员列表与核对过的原包不同")
        raw = source.read("iris.data")
    if sha256(raw) != RAW_SHA256:
        raise ValueError("iris.data 的字节身份发生变化")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and RAW_FILE.read_bytes() != raw:
        raise ValueError("已有 iris.data 与 UCI 原包不同，不覆盖")
    if not RAW_FILE.exists():
        RAW_FILE.write_bytes(raw)
    return raw


def parse_rows(raw):
    rows = []
    for row_id, cells in enumerate(csv.reader(io.StringIO(raw.decode("ascii"))), start=1):
        if not cells:
            continue
        if len(cells) != 5 or cells[4] not in CLASS_TO_LABEL:
            raise ValueError(f"第 {row_id} 行不符合四项测量加类别的格式")
        values = [float(item) for item in cells[:4]]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"第 {row_id} 行的测量不是有限正数")
        rows.append({
            "source_row": row_id,
            "all_four_cm": values,
            "petal_cm": values[2:4],
            "species": cells[4],
            "label": CLASS_TO_LABEL[cells[4]],
        })
    if len(rows) != 150:
        raise ValueError(f"期望 150 行，实际 {len(rows)} 行")
    if Counter(row["species"] for row in rows) != {name: 50 for name in CLASS_TO_LABEL}:
        raise ValueError("三类样本数不符合原包说明")
    return rows


def assign_groups(rows):
    splits = {part: [] for part in QUOTAS}
    duplicate_groups = []
    for species, label in (("Iris-versicolor", -1), ("Iris-virginica", 1)):
        by_measurement = defaultdict(list)
        for row in rows:
            if row["species"] == species:
                by_measurement[tuple(row["all_four_cm"])].append(row)
        groups = sorted(by_measurement.values(), key=lambda group: group[0]["source_row"])
        duplicate_groups.extend(
            [row["source_row"] for row in group]
            for group in groups if len(group) > 1
        )
        random.Random(SEED).shuffle(groups)
        assigned = {part: [] for part in QUOTAS}
        for group in groups:
            for part, quota in QUOTAS.items():
                if len(assigned[part]) + len(group) <= quota:
                    assigned[part].extend(group)
                    break
            else:
                raise ValueError(f"{species} 中有一组行无法按预定容量分配")
        if any(len(assigned[part]) != quota for part, quota in QUOTAS.items()):
            raise ValueError(f"{species} 的切分没有达到预定的 30/10/10")
        for part in QUOTAS:
            splits[part].extend(assigned[part])
    for part in splits:
        splits[part].sort(key=lambda row: row["source_row"])
    if set(row["source_row"] for part in splits.values() for row in part) != {
        row["source_row"] for row in rows if row["label"] is not None
    }:
        raise ValueError("目标两类有行缺失或重复分配")
    membership = {
        row["source_row"]: part
        for part, records in splits.items() for row in records
    }
    for source_rows in duplicate_groups:
        if len({membership[row_id] for row_id in source_rows}) != 1:
            raise ValueError("同一组四项测量跨越切分")
    return splits, duplicate_groups


def main():
    rows = parse_rows(load_raw())
    splits, duplicate_groups = assign_groups(rows)
    report = {
        "protocol": "work/verification/C08_iris_protocol.md",
        "source": "work/data/uci_iris_SOURCE.md",
        "archive_sha256": ARCHIVE_SHA256,
        "iris_data_sha256": RAW_SHA256,
        "seed": SEED,
        "target_species": {"Iris-versicolor": -1, "Iris-virginica": 1},
        "feature_columns_zero_based": [2, 3],
        "feature_names": ["petal_length_cm", "petal_width_cm"],
        "duplicate_all_four_groups_source_rows": duplicate_groups,
        "splits": {
            part: [
                {"source_row": row["source_row"], "petal_cm": row["petal_cm"],
                 "label": row["label"], "species": row["species"]}
                for row in records
            ]
            for part, records in splits.items()
        },
    }
    SPLIT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("UCI iris.data 原包与切分已核对")
    for part, records in splits.items():
        print(part, len(records), dict(Counter(row["species"] for row in records)))
    print("四项数值完全相同的组（目标两类）:", duplicate_groups)


if __name__ == "__main__":
    main()
