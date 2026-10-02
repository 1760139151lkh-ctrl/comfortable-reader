"""Train a source-bound UCI Adult model, then open its original test once.

This is a historical benchmark exercise, never an employment or lending rule.
The model never uses sex/race as features; sex is retained for audit and for
an explicitly separate demonstration of group-dependent thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
DATA = WORK / "data/c44_adult"
SEED = 20260925
COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "marital-status", "occupation", "relationship", "native-country"]
FEATURES = NUM + CAT
FILE_SHA = {"adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
            "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(name: str, expected: int) -> pd.DataFrame:
    path = DATA / name
    if sha(path) != FILE_SHA[name]:
        raise RuntimeError(f"UCI file changed: {name}")
    rows = []
    with path.open(encoding="ascii", newline="") as source:
        for line in csv.reader(source):
            if not line or (len(line) == 1 and line[0].startswith("|")):
                continue
            row = [part.strip() for part in line]
            if len(row) != 15:
                raise RuntimeError(f"unexpected UCI field count: {len(row)}")
            row[-1] = row[-1].removesuffix(".")
            if row[-1] not in ("<=50K", ">50K") or row[9] not in ("Male", "Female"):
                raise RuntimeError("unexpected official label or group encoding")
            rows.append(row)
    if len(rows) != expected:
        raise RuntimeError(f"unexpected UCI count {name}: {len(rows)}")
    table = pd.DataFrame(rows, columns=COLS)
    for column in NUM:
        table[column] = pd.to_numeric(table[column], errors="raise")
    return table


def y_of(table: pd.DataFrame) -> np.ndarray:
    return (table["income"].to_numpy() == ">50K").astype(np.int64)


def summary(y: np.ndarray, prob: np.ndarray, group: np.ndarray, thresholds: dict[str, float]):
    result = {}
    for label in sorted(set(group)):
        mask = group == label
        actual = y[mask]
        predicted = prob[mask] >= thresholds[label]
        tp = int(np.sum(predicted & (actual == 1)))
        fp = int(np.sum(predicted & (actual == 0)))
        tn = int(np.sum(~predicted & (actual == 0)))
        fn = int(np.sum(~predicted & (actual == 1)))
        n = len(actual)
        result[label] = {"n": n, "actual_positive": int(actual.sum()),
                         "label_rate": float(actual.mean()), "threshold": float(thresholds[label]),
                         "predicted_positive": int(predicted.sum()),
                         "selection_rate": float(predicted.mean()),
                         "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                         "tpr_given_label_positive": tp / (tp + fn) if tp + fn else None,
                         "fpr_given_label_negative": fp / (fp + tn) if fp + tn else None,
                         "ppv_given_predicted_positive": tp / (tp + fp) if tp + fp else None,
                         "accuracy": (tp + tn) / n}
    predicted_all = np.array([prob[i] >= thresholds[group[i]] for i in range(len(y))])
    return {"overall": {"n": len(y), "accuracy": float((predicted_all == y).mean()),
                         "predicted_positive": int(predicted_all.sum()),
                         "selection_rate": float(predicted_all.mean()),
                         "nll_nats_unchanged_by_threshold": float(log_loss(y, prob, labels=[0, 1])),
                         "roc_auc_unchanged_by_threshold": float(roc_auc_score(y, prob))},
            "by_group": result}


def preprocess(train: pd.DataFrame):
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), NUM),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ], remainder="drop", sparse_threshold=0.0)
    transformer.fit(train[FEATURES])
    return transformer


def train(out_dir: Path):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new run directory")
    table = read_records("adult.data", 32561)  # Do not parse adult.test here.
    y = y_of(table)
    sex = table["sex"].to_numpy()
    stratify = np.array([f"{sex[i]}/{y[i]}" for i in range(len(y))])
    fit_ids, val_ids = train_test_split(np.arange(len(y)), test_size=0.20,
                                        random_state=SEED, stratify=stratify)
    transformer = preprocess(table.iloc[fit_ids])
    fit_x = transformer.transform(table.iloc[fit_ids][FEATURES])
    val_x = transformer.transform(table.iloc[val_ids][FEATURES])
    fit_y, val_y = y[fit_ids], y[val_ids]
    if not np.isfinite(fit_x).all() or not np.isfinite(val_x).all():
        raise RuntimeError("non-finite features")
    candidates = []
    best = None
    for C in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=C, max_iter=1500, solver="lbfgs", random_state=SEED)
        model.fit(fit_x, fit_y)
        probability = model.predict_proba(val_x)[:, 1]
        loss = float(log_loss(val_y, probability, labels=[0, 1]))
        candidates.append({"C": C, "validation_nll_nats": loss,
                           "validation_accuracy_at_0_5": float(((probability >= 0.5) == val_y).mean()),
                           "n_iter": int(model.n_iter_[0])})
        if best is None or (loss, C) < (best[0], best[1]):
            best = (loss, C, model, probability)
    _, chosen_C, chosen_model, val_prob = best
    thresholds = {}
    val_sex = sex[val_ids]
    for group in ("Female", "Male"):
        positive_scores = val_prob[(val_sex == group) & (val_y == 1)]
        if len(positive_scores) < 100:
            raise RuntimeError("positive validation group too small for planned quantile")
        thresholds[group] = float(np.quantile(positive_scores, 0.2, method="higher"))
    out.mkdir(parents=True)
    split = out / "split.npz"
    np.savez_compressed(split, fit_indices=fit_ids, validation_indices=val_ids)
    model_file = out / "selected_model.joblib"
    joblib.dump({"preprocessor": transformer, "classifier": chosen_model,
                 "features": FEATURES, "C": chosen_C}, model_file)
    baseline = {"Female": 0.5, "Male": 0.5}
    report = {"source_train_sha256": FILE_SHA["adult.data"], "script_sha256": sha(Path(__file__)),
              "split_seed": SEED, "split_file": "split.npz", "split_sha256": sha(split),
              "fit_rows": len(fit_ids), "validation_rows": len(val_ids),
              "feature_columns": FEATURES, "excluded_from_model": ["sex", "race", "fnlwgt", "education"],
              "missing_as_category": "?", "candidates": candidates,
              "selected_C_by_validation_nll": chosen_C,
              "selected_model_file": "selected_model.joblib", "selected_model_sha256": sha(model_file),
              "thresholds_validation_TPR_target_0_8": thresholds,
              "validation_global_threshold_0_5": summary(val_y, val_prob, val_sex, baseline),
              "validation_group_TPR_target_0_8": summary(val_y, val_prob, val_sex, thresholds),
              "official_test_status": "not_parsed_or_scored_in_train_mode",
              "scope": "1994 filtered UCI Adult source rows, no fnlwgt population weighting, logistic income-label task not a personnel decision; sex/race excluded as model inputs, proxies remain possible"}
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fit": len(fit_ids), "validation": len(val_ids),
                      "chosen_C": chosen_C, "thresholds": thresholds,
                      "val_baseline": report["validation_global_threshold_0_5"]["overall"],
                      "val_group_threshold": report["validation_group_TPR_target_0_8"]["overall"]}, ensure_ascii=False))


def test(run_dir: Path):
    out = run_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or (out / "test.json").exists():
        raise ValueError("run directory must exist in work/runs and test not be opened")
    train_report = json.loads((out / "train.json").read_text(encoding="utf-8"))
    if train_report["script_sha256"] != sha(Path(__file__)) or train_report["split_sha256"] != sha(out / "split.npz"):
        raise RuntimeError("training code or data split changed before test")
    model_path = out / train_report["selected_model_file"]
    if train_report["selected_model_sha256"] != sha(model_path):
        raise RuntimeError("selected model bytes changed before test")
    bundle = joblib.load(model_path)
    if bundle["C"] != train_report["selected_C_by_validation_nll"] or bundle["features"] != FEATURES:
        raise RuntimeError("unexpected model metadata")
    table = read_records("adult.test", 16281)  # First semantic open for this chapter.
    y = y_of(table)
    sex = table["sex"].to_numpy()
    race = table["race"].to_numpy()
    x = bundle["preprocessor"].transform(table[FEATURES])
    prob = bundle["classifier"].predict_proba(x)[:, 1]
    baseline = {"Female": 0.5, "Male": 0.5}
    thresholds = train_report["thresholds_validation_TPR_target_0_8"]
    race_group = summary(y, prob, race, {key: 0.5 for key in sorted(set(race))})
    report = {"training_report_sha256": sha(out / "train.json"),
              "official_test_source_sha256": FILE_SHA["adult.test"],
              "first_chapter_test_after_model_and_threshold_selection": True,
              "test_rows": len(y), "positive_label": int(y.sum()),
              "global_threshold_0_5": summary(y, prob, sex, baseline),
              "validation_chosen_group_thresholds": summary(y, prob, sex, thresholds),
              "baseline_race_coded_groups_descriptive_only": race_group,
              "scope": "original 1994 source random test split, row-weighted; label and sex/race categories as in file, no individual decisions or population/causal fairness inference"}
    (out / "test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test_rows": len(y),
                      "global": report["global_threshold_0_5"]["overall"],
                      "group_threshold": report["validation_chosen_group_thresholds"]["overall"],
                      "sex_group": {key: {k: val[k] for k in ("n", "label_rate", "selection_rate", "tpr_given_label_positive", "fpr_given_label_negative", "ppv_given_predicted_positive")}
                                    for key, val in report["global_threshold_0_5"]["by_group"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("train").add_argument("--run-dir", type=Path, required=True)
    sub.add_parser("test").add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    train(args.run_dir) if args.mode == "train" else test(args.run_dir)
