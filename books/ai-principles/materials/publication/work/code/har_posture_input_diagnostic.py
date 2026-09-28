"""事后输入诊断：SITTING/STANDING 的 body_acc 与 total_acc 窗口均值。"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data" / "uci_har"
RESULT = WORK / "results" / "har_posture_input_diagnostic.json"
FIGURE = WORK / "figures" / "har_posture_input_diagnostic.png"


def read_total_acc(part: str) -> np.ndarray:
    with zipfile.ZipFile(DATA / "UCI_HAR_Dataset.zip") as outer:
        inner_bytes = outer.read("UCI HAR Dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        arrays = []
        for axis in "xyz":
            member = f"UCI HAR Dataset/{part}/Inertial Signals/total_acc_{axis}_{part}.txt"
            arrays.append(np.loadtxt(io.BytesIO(inner.read(member)), dtype=np.float32))
    return np.stack(arrays, axis=-1)


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    predicted = model.predict(x)
    matrix = confusion_matrix(y, predicted, labels=[3, 4])
    return {
        "correct": int(np.trace(matrix)),
        "total": len(y),
        "accuracy": float(np.trace(matrix) / len(y)),
        "confusion_true_sitting_standing": matrix.tolist(),
    }


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((DATA / "UCI_HAR_Dataset.zip").read_bytes()).hexdigest() != manifest["outer_zip_sha256"]:
        raise ValueError("官方包与主实验身份不符")
    with np.load(DATA / "har_sequences.npz") as ready:
        raw = {key: ready[key].copy() for key in ready.files}
    train_body = raw["train_x"][:, :, :3].mean(axis=1)
    test_body = raw["test_x"][:, :, :3].mean(axis=1)
    train_total = read_total_acc("train").mean(axis=1)
    test_total = read_total_acc("test").mean(axis=1)
    if train_total.shape != train_body.shape or test_total.shape != test_body.shape:
        raise ValueError("total_acc 与主实验窗口行号／三轴形状不符")
    train_rows = raw["train_indices"]
    valid_rows = raw["validation_indices"]
    train_rows = train_rows[np.isin(raw["train_y"][train_rows], [3, 4])]
    valid_rows = valid_rows[np.isin(raw["train_y"][valid_rows], [3, 4])]
    test_rows = np.flatnonzero(np.isin(raw["test_y"], [3, 4]))
    conditions = {}
    for name, train_features, test_features in (
        ("body_acc_mean", train_body, test_body),
        ("total_acc_mean", train_total, test_total),
    ):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))
        model.fit(train_features[train_rows], raw["train_y"][train_rows])
        conditions[name] = {
            "train": evaluate(model, train_features[train_rows], raw["train_y"][train_rows]),
            "validation": evaluate(model, train_features[valid_rows], raw["train_y"][valid_rows]),
            "official_test_exploratory": evaluate(model, test_features[test_rows], raw["test_y"][test_rows]),
            "train_class_mean_sitting": train_features[train_rows][raw["train_y"][train_rows] == 3].mean(axis=0).tolist(),
            "train_class_mean_standing": train_features[train_rows][raw["train_y"][train_rows] == 4].mean(axis=0).tolist(),
        }
    record = {
        "identity": "post-run exploratory input audit after seeing six-channel HAR posture errors; does not revise four-arm comparison",
        "source_outer_sha256": manifest["outer_zip_sha256"],
        "training_subject_ids": manifest["train_subject_ids"],
        "validation_subject_ids": manifest["validation_subject_ids"],
        "official_test_subject_ids": manifest["official_test_subject_ids"],
        "classes_zero_based": {"3": "SITTING", "4": "STANDING"},
        "feature": "mean of each of three axes across the official 128-step preprocessed window",
        "classifier": "StandardScaler from training subjects + LogisticRegression C=1, lbfgs",
        "conditions": conditions,
    }
    RESULT.parent.mkdir(exist_ok=True)
    FIGURE.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5.2, 3.1), layout="constrained")
    places = np.arange(2)
    for offset, split, label in ((-0.18, "validation", "validation subjects"), (0.18, "official_test_exploratory", "official test subjects")):
        values = [conditions[condition][split]["accuracy"] for condition in ("body_acc_mean", "total_acc_mean")]
        ax.bar(places + offset, values, width=0.34, label=label)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set(xticks=places, xticklabels=["body acceleration", "total acceleration"],
           ylim=(0, 1), ylabel="SITTING vs STANDING accuracy",
           title="post-run input diagnostic")
    ax.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print(json.dumps({name: {split: data[split]["accuracy"] for split in ("validation", "official_test_exploratory")}
                      for name, data in conditions.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
