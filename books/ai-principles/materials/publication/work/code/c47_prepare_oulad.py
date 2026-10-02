"""Build time-limited BBB OULAD snapshots without exposing held-out labels."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/c47_oulad"
OUT = DATA / "snapshots_v1"
ARCHIVE_SHA = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"
PRESENTATIONS = ("2013B", "2013J", "2014B", "2014J")
HORIZONS = (7, 21, 42)
OUTCOMES = {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if source["sha256"] != ARCHIVE_SHA:
        raise RuntimeError("Original UCI archive identity changed")
    for name, member in source["members"].items():
        if sha(DATA / name) != member["sha256"]:
            raise RuntimeError(f"UCI CSV bytes changed: {name}")
    info = pd.read_csv(DATA / "studentInfo.csv",
                       usecols=["code_module", "code_presentation", "id_student", "final_result"])
    info = info[(info.code_module == "BBB") & info.code_presentation.isin(PRESENTATIONS)].copy()
    info["presentation_order"] = info.code_presentation.map({name: i for i, name in enumerate(PRESENTATIONS)})
    info = info.sort_values(["presentation_order", "id_student"]).drop_duplicates("id_student", keep="first")
    info["label"] = info.final_result.map(OUTCOMES)
    if info.label.isna().any():
        raise ValueError("Unknown final_result category")
    registration = pd.read_csv(DATA / "studentRegistration.csv")
    registration = registration[(registration.code_module == "BBB")
                                & registration.code_presentation.isin(PRESENTATIONS)].copy()
    registration["registration_day"] = pd.to_numeric(registration.date_registration, errors="coerce")
    registration["unregistration_day"] = pd.to_numeric(registration.date_unregistration, errors="coerce")
    info = info.merge(
        registration[["code_module", "code_presentation", "id_student", "registration_day", "unregistration_day"]],
        on=["code_module", "code_presentation", "id_student"], how="left", validate="one_to_one")
    if len(info) == 0 or info.registration_day.isna().all():
        raise RuntimeError("Registration linkage failed")
    weekly = defaultdict(lambda: np.zeros(6, dtype=np.int64))
    scanned = 0
    included = 0
    for chunk in pd.read_csv(
        DATA / "studentVle.csv",
        usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"],
        chunksize=500_000,
    ):
        scanned += len(chunk)
        part = chunk[(chunk.code_module == "BBB") & chunk.code_presentation.isin(PRESENTATIONS)
                     & (chunk.date >= 0) & (chunk.date < 42)].copy()
        if part.empty:
            continue
        clicks = pd.to_numeric(part.sum_click, errors="coerce")
        if clicks.isna().any() or (clicks < 0).any():
            raise ValueError("Invalid observed click count")
        part["sum_click"] = clicks.astype(np.int64)
        part["week"] = (part.date // 7).astype(int)
        grouped = part.groupby(["code_presentation", "id_student", "week"], sort=False)["sum_click"].sum()
        for (presentation, student, week), count in grouped.items():
            weekly[(str(presentation), int(student))][int(week)] += int(count)
        included += len(part)
    if scanned != 10_655_280:
        raise RuntimeError(f"Unexpected original VLE row count: {scanned}")
    manifest = {
        "source_manifest_sha256": sha(DATA / "manifest.json"),
        "task": "BBB final Pass/Distinction vs Fail/Withdrawn from past VLE weekly click sums",
        "presentations": {"train": ["2013B", "2013J"], "validation": ["2014B"], "test": ["2014J"]},
        "same_anonymous_student_later_presentations_removed": True,
        "scanned_vle_rows": scanned, "vle_rows_BBB_day0_to41": included,
        "horizons": {},
        "test_label_content_not_printed": True,
    }
    for horizon in HORIZONS:
        eligible = info[(info.registration_day <= horizon)
                        & (info.unregistration_day.isna() | (info.unregistration_day > horizon))]
        group_report = {}
        for split, presentations in manifest["presentations"].items():
            rows = eligible[eligible.code_presentation.isin(presentations)].sort_values(
                ["presentation_order", "id_student"])
            features = np.array([
                weekly[(str(row.code_presentation), int(row.id_student))][:horizon // 7]
                for row in rows.itertuples(index=False)
            ], dtype=np.int64)
            labels = rows.label.to_numpy(dtype=np.int8)
            if features.ndim != 2 or features.shape[0] != len(labels):
                raise RuntimeError(f"Snapshot shape mismatch: {split}, {horizon}")
            x_path = OUT / f"{split}_day{horizon}_X.npy"
            y_path = OUT / f"{split}_day{horizon}_y.npy"
            np.save(x_path, features, allow_pickle=False)
            np.save(y_path, labels, allow_pickle=False)
            group_report[split] = {
                "n": len(labels), "weekly_shape": list(features.shape),
                "X_sha256": sha(x_path), "y_sha256": sha(y_path),
                "positive": int(labels.sum()) if split == "train" else "sealed_until_test",
            }
        manifest["horizons"][str(horizon)] = group_report
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
