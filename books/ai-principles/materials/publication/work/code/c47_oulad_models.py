"""Past-only OULAD BBB risk prediction, then one locked later-presentation score."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/c47_oulad/snapshots_v1"
HORIZONS = (7, 21, 42)
C_GRID = (0.01, 0.1, 1.0, 10.0)
SEED = 4701


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_manifest(*, include_test: bool) -> tuple[dict, str]:
    p = DATA / "manifest.json"
    manifest = json.loads(p.read_text(encoding="utf-8"))
    for horizon in HORIZONS:
        splits = ("train", "validation", "test") if include_test else ("train", "validation")
        for split in splits:
            row = manifest["horizons"][str(horizon)][split]
            for suffix in ("X", "y"):
                f = DATA / f"{split}_day{horizon}_{suffix}.npy"
                if sha(f) != row[f"{suffix}_sha256"]:
                    raise RuntimeError(f"Snapshot bytes changed: {f}")
    return manifest, sha(p)


def load(horizon: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(DATA / f"{split}_day{horizon}_X.npy", allow_pickle=False)
    y = np.load(DATA / f"{split}_day{horizon}_y.npy", allow_pickle=False)
    if len(x) != len(y) or x.shape[1] != horizon // 7 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Bad snapshot {horizon}/{split}")
    return x, y


def features(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "total":
        return np.log1p(x.sum(axis=1, keepdims=True)).astype(np.float64)
    return np.log1p(x).astype(np.float64)


def fit_logistic(xtrain: np.ndarray, ytrain: np.ndarray, xval: np.ndarray, yval: np.ndarray, kind: str) -> tuple[dict, list]:
    a = features(xtrain, kind)
    b = features(xval, kind)
    scaler = StandardScaler()
    a = scaler.fit_transform(a)
    b = scaler.transform(b)
    choices = []
    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", tol=1e-8)
        model.fit(a, ytrain)
        probability = model.predict_proba(b)[:, 1]
        score = float(log_loss(yval, probability, labels=[0, 1]))
        choices.append({"C": c, "validation_nll": score})
        if best is None or score < best[0]:
            best = (score, c, model)
    chosen = best[2]
    record = {
        "kind": kind, "C": best[1], "validation_nll": best[0],
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficients": chosen.coef_[0].tolist(), "intercept": float(chosen.intercept_[0]),
    }
    return record, choices


def logistic_predict(x: np.ndarray, record: dict) -> np.ndarray:
    a = features(x, record["kind"])
    mean = np.array(record["scaler_mean"])
    scale = np.array(record["scaler_scale"])
    coef = np.array(record["coefficients"])
    z = ((a - mean) / scale) @ coef + record["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class WeekGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, last = self.gru(x)
        return self.head(last[-1]).squeeze(-1)


def gru_input(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(np.log1p(x.astype(np.float32))[:, :, None] / 5.0, device=device)


def gru_prob(model: WeekGRU, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            output = model(gru_input(x[start:start+256], device))
            outputs.append(torch.sigmoid(output).cpu().numpy())
    return np.concatenate(outputs)


def fit_gru(xtrain: np.ndarray, ytrain: np.ndarray, xval: np.ndarray, yval: np.ndarray,
            checkpoint: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = WeekGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    target = torch.tensor(ytrain, dtype=torch.float32, device=device)
    xx = gru_input(xtrain, device)
    history = []
    best = None
    for epoch in range(1, 21):
        model.train()
        order = torch.randperm(len(xtrain), device=device)
        for selection in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx[selection])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target[selection])
            loss.backward()
            optimizer.step()
        probability = gru_prob(model, xval, device)
        nll = float(log_loss(yval, probability, labels=[0, 1]))
        history.append({"epoch": epoch, "validation_nll": nll})
        if best is None or nll < best[0]:
            best = (nll, epoch, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
    torch.save({"state": best[2], "chosen_epoch": best[1], "seed": SEED}, checkpoint)
    return {"chosen_epoch": best[1], "validation_nll": best[0], "history": history,
            "device": device, "checkpoint_sha256": sha(checkpoint),
            "parameter_count": sum(p.numel() for p in model.parameters())}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0}
    p = np.clip(p, 1e-8, 1 - 1e-8)
    answer = {"n": int(len(y)), "positive": int(y.sum()),
              "positive_rate": float(np.mean(y)), "nll": float(log_loss(y, p, labels=[0, 1])),
              "brier": float(brier_score_loss(y, p)), "accuracy_threshold_0_5": float(accuracy_score(y, p >= .5))}
    answer["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return answer


def train(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha = checked_manifest(include_test=False)
    result = {"scope": "BBB 2013B/J train and 2014B validation only; 2014J labels not opened in training mode",
              "manifest_sha256": manifest_sha, "code_sha256": sha(Path(__file__)),
              "logistic_candidates_C": list(C_GRID), "horizons": {}, "gru_day42": None}
    fitted = {}
    for horizon in HORIZONS:
        train_x, train_y = load(horizon, "train")
        val_x, val_y = load(horizon, "validation")
        baseline = float(np.mean(train_y))
        row = {"train_n": len(train_y), "train_positive": int(train_y.sum()),
               "validation_n": len(val_y), "validation_positive": int(val_y.sum()),
               "baseline_positive_probability": baseline,
               "baseline_validation": metrics(val_y, np.full(len(val_y), baseline)),
               "models": {}, "candidate_validation": {}}
        for kind in ("total", "weekly"):
            record, choices = fit_logistic(train_x, train_y, val_x, val_y, kind)
            row["models"][kind] = record
            row["candidate_validation"][kind] = choices
            row[kind + "_validation"] = metrics(val_y, logistic_predict(val_x, record))
        result["horizons"][str(horizon)] = row
        fitted[str(horizon)] = {"baseline": baseline, "total": row["models"]["total"],
                                 "weekly": row["models"]["weekly"]}
    checkpoint = run_dir / "gru_day42.pt"
    xtrain, ytrain = load(42, "train")
    xval, yval = load(42, "validation")
    result["gru_day42"] = fit_gru(xtrain, ytrain, xval, yval, checkpoint)
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = WeekGRU().to(result["gru_day42"]["device"])
    model.load_state_dict(selected["state"])
    result["gru_day42"]["selected_validation"] = metrics(yval, gru_prob(model, xval, result["gru_day42"]["device"]))
    model_path = run_dir / "models.json"
    model_path.write_text(json.dumps(fitted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["models_sha256"] = sha(model_path)
    (run_dir / "train.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_sha256": manifest_sha,
                      "selected_C": {h: {k: v["C"] for k, v in row["models"].items()}
                                     for h, row in result["horizons"].items()},
                      "validation_nll": {h: {"baseline": row["baseline_validation"]["nll"],
                                              "total": row["total_validation"]["nll"],
                                              "weekly": row["weekly_validation"]["nll"]}
                                         for h, row in result["horizons"].items()},
                      "gru_epoch": result["gru_day42"]["chosen_epoch"],
                      "gru_validation_nll": result["gru_day42"]["validation_nll"]}, ensure_ascii=False, indent=2))


def test(run_dir: Path) -> None:
    result_path = run_dir / "test.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    manifest, manifest_sha = checked_manifest(include_test=True)
    train_path = run_dir / "train.json"
    report = json.loads(train_path.read_text(encoding="utf-8"))
    if report["manifest_sha256"] != manifest_sha or report["code_sha256"] != sha(Path(__file__)):
        raise RuntimeError("Training source/code changed before test")
    model_path = run_dir / "models.json"
    if sha(model_path) != report["models_sha256"]:
        raise RuntimeError("Chosen logistic models changed")
    models = json.loads(model_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / "gru_day42.pt"
    if sha(checkpoint) != report["gru_day42"]["checkpoint_sha256"]:
        raise RuntimeError("Chosen GRU changed")
    result = {"scope": "First single-mode 2014J later-presentation scoring after all horizons and models fixed; course BBB, new anonymous student IDs and only eligible snapshot rows",
              "training_report_sha256": sha(train_path), "manifest_sha256": manifest_sha,
              "horizons": {}, "gru_day42": None}
    for horizon in HORIZONS:
        x, y = load(horizon, "test")
        fixed = models[str(horizon)]
        row = {"baseline": metrics(y, np.full(len(y), fixed["baseline"])),
               "total": metrics(y, logistic_predict(x, fixed["total"])),
               "weekly": metrics(y, logistic_predict(x, fixed["weekly"]))}
        if horizon == 42:
            zero = x.sum(axis=1) == 0
            prediction = logistic_predict(x, fixed["weekly"])
            row["no_click_first_42_days"] = metrics(y[zero], prediction[zero])
            row["some_click_first_42_days"] = metrics(y[~zero], prediction[~zero])
        result["horizons"][str(horizon)] = row
    x42, y42 = load(42, "test")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = WeekGRU().to(device)
    net.load_state_dict(saved["state"])
    result["gru_day42"] = metrics(y42, gru_prob(net, x42, device))
    (run_dir / "test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    days = list(HORIZONS)
    for kind, color in (("baseline", "#858585"), ("total", "#bc7039"), ("weekly", "#236e87")):
        axes[0].plot(days, [result["horizons"][str(h)][kind]["nll"] for h in days],
                     marker="o", color=color, label=kind)
    axes[0].scatter([42], [result["gru_day42"]["nll"]], color="#824793", marker="s", label="GRU (42 days)")
    axes[0].set(xlabel="observation cutoff day", ylabel="later-cohort log loss",
                title="Prediction among still-registered students")
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    weights = np.array(models["42"]["weekly"]["coefficients"])
    axes[1].bar(np.arange(1, 7), weights, color="#236e87")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="week of recorded clicks", ylabel="standardized log-click coefficient",
                title="Chosen 42-day weekly logistic model")
    axes[1].grid(axis="y", alpha=.2)
    png = run_dir / "bbb_early_clicks_later_course_result.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)
    receipt = {"test_sha256": sha(result_path), "figure_sha256": sha(png),
               "figure": str(png.relative_to(ROOT)).replace("\\", "/")}
    (run_dir / "figure_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": {h: {k: round(v["nll"], 5) for k, v in result["horizons"][str(h)].items()
                                     if k in ("baseline", "total", "weekly")}
                                for h in HORIZONS},
                      "test_n": {h: result["horizons"][str(h)]["baseline"]["n"] for h in HORIZONS},
                      "gru_day42_nll": result["gru_day42"]["nll"],
                      "figure_sha256": receipt["figure_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("train", "test"))
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    (train if args.stage == "train" else test)(ROOT / args.run_dir)
