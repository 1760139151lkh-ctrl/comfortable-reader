"""A public-data count mechanism and a local-gradient identity.

The process owns all rows. It demonstrates DP mathematics on a public source
and federated averaging algebra, not private training or remote clients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from c44_adult_group_study import FEATURES, FILE_SHA, RUNS, read_records, y_of


WORK = Path(__file__).resolve().parents[1]
SEED = 20260925


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x.T @ (expit(x @ w) - y) / len(y)


def loss(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    z = x @ w
    return float(np.mean(np.logaddexp(0, z) - y * z))


def main(run_dir: Path, out_path: Path):
    run = run_dir.resolve()
    out = out_path.resolve()
    if not run.is_relative_to(RUNS.resolve()) or not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("use existing C44 training directory and new work/runs JSON")
    record = json.loads((run / "train.json").read_text(encoding="utf-8"))
    if record["source_train_sha256"] != FILE_SHA["adult.data"] or sha(run / "split.npz") != record["split_sha256"]:
        raise RuntimeError("C44 train source or split differs")
    model_file = run / record["selected_model_file"]
    if sha(model_file) != record["selected_model_sha256"]:
        raise RuntimeError("C44 preprocessing model differs")
    bundle = joblib.load(model_file)
    table = read_records("adult.data", 32561)  # Never load adult.test here.
    with np.load(run / "split.npz") as saved:
        fit_ids = saved["fit_indices"].astype(np.int64)
    x = np.asarray(bundle["preprocessor"].transform(table.iloc[fit_ids][FEATURES]), dtype=np.float64)
    x = np.column_stack((np.ones(len(x)), x))
    y = y_of(table)[fit_ids].astype(np.float64)
    sex = table["sex"].to_numpy()[fit_ids]
    names = ["Female/<=50K", "Female/>50K", "Male/<=50K", "Male/>50K"]
    counts = np.array([np.sum((sex == group) & (y == label)) for group in ("Female", "Male") for label in (0, 1)], dtype=np.int64)
    if counts.sum() != len(fit_ids) or (counts <= 0).any():
        raise RuntimeError("unexpected training group counts")
    neighbor = counts.copy()
    neighbor[1] -= 1  # remove one Female/>50K row under add/remove adjacency
    if int(np.abs(counts - neighbor).sum()) != 1:
        raise RuntimeError("unexpected L1 sensitivity witness")
    rng = np.random.default_rng(SEED)
    releases = {}
    for eps in (0.5, 2.0):
        scale = 1 / eps
        noisy = counts + rng.laplace(0.0, scale, size=4)
        log_ratio = (np.abs(noisy - neighbor).sum() - np.abs(noisy - counts).sum()) / scale
        if abs(log_ratio) > eps + 1e-10:
            raise RuntimeError("one-point density-ratio check outside derived bound")
        releases[str(eps)] = {"epsilon_add_remove_one": eps, "laplace_scale_per_cell": scale,
                              "noisy_cells": noisy.tolist(),
                              "observed_log_density_ratio_to_one_removed_neighbor": float(log_ratio),
                              "proof_scope": "all possible outputs via reverse triangle inequality; observed output only a numerical check"}
    clients = {}
    pieces = []
    for group in ("Female", "Male"):
        mask = sex == group
        xi, yi = x[mask], y[mask]
        clients[group] = {"rows": len(yi), "source_label_rate": float(yi.mean())}
        pieces.append((xi, yi))
    eta = 0.2
    rounds = 5
    zero = np.zeros(x.shape[1], dtype=np.float64)
    central_gradient = gradient(x, y, zero)
    weighted_client_gradient = sum(len(yi) / len(y) * gradient(xi, yi, zero) for xi, yi in pieces)
    central_one = zero - eta * central_gradient
    local_one = sum(len(yi) / len(y) * (zero - eta * gradient(xi, yi, zero)) for xi, yi in pieces)
    if np.max(np.abs(central_one - local_one)) > 1e-12:
        raise RuntimeError("one local gradient step should equal centralized full batch step")
    central_many = zero.copy()
    for _ in range(rounds):
        central_many -= eta * gradient(x, y, central_many)
    local_models = []
    for xi, yi in pieces:
        wi = zero.copy()
        for _ in range(rounds):
            wi -= eta * gradient(xi, yi, wi)
        local_models.append((len(yi) / len(y), wi))
    fed_average = sum(weight * wi for weight, wi in local_models)
    report = {"source_train_sha256": FILE_SHA["adult.data"],
              "c44_training_report_sha256": sha(run / "train.json"),
              "source_split_sha256": record["split_sha256"],
              "source_model_sha256": record["selected_model_sha256"],
              "script_sha256": sha(Path(__file__)),
              "input_scope": "only C44 26048 fit rows from public UCI Adult; same processed features as selected logistic model but separate no-L2 teaching gradient",
              "four_cells": {"order": names, "exact_public_source_counts": counts.tolist(),
                             "add_remove_one_l1_sensitivity": 1,
                             "replacement_one_l1_sensitivity": 2,
                             "two_releases_combined_epsilon_if_source_were_private": 2.5,
                             "real_world_boundary": "original UCI rows and exact counts are already public here; adding noise to this copy does not make them private"},
              "noisy_releases": releases,
              "two_simulated_clients": clients,
              "same_initial_weights_zero": True,
              "full_batch_logistic_step_size": eta,
              "one_step_gradient_max_abs_difference": float(np.max(np.abs(central_gradient - weighted_client_gradient))),
              "one_step_parameter_max_abs_difference": float(np.max(np.abs(central_one - local_one))),
              "five_local_vs_five_central_weight_l2": float(np.linalg.norm(fed_average - central_many)),
              "five_central_train_loss": loss(x, y, central_many),
              "five_local_average_train_loss": loss(x, y, fed_average),
              "limit": "same Python process owns both clients and all raw records; no secure aggregation, no DP-trained model, no independent devices or communication measurement"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts.tolist(),
                      "one_step_difference": report["one_step_parameter_max_abs_difference"],
                      "five_step_l2": report["five_local_vs_five_central_weight_l2"],
                      "client_rows": clients}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run_dir, args.out)
