"""One-star Kepler Q1 signal fit and locked Q2 time-forward check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares, LombScargle

DATA = Path(__file__).resolve().parents[1] / "data" / "c45_kepler_hatp7"
FILES = {
    "q1": DATA / "kplr010666592-2009166043257_llc.fits",
    "q2": DATA / "kplr010666592-2009259160929_llc.fits",
}
DURATIONS = np.array([0.10, 0.14, 0.18, 0.22], dtype=float)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    path = FILES[which]
    with fits.open(path, memmap=False) as hdul:
        h0 = hdul[0].header
        h1 = hdul[1].header
        data = hdul[1].data
        t = np.array(data["TIME"], dtype=np.float64)
        f = np.array(data["PDCSAP_FLUX"], dtype=np.float64)
        e = np.array(data["PDCSAP_FLUX_ERR"], dtype=np.float64)
        q = np.array(data["SAP_QUALITY"], dtype=np.int64)
        metadata = {"quarter": int(h0["QUARTER"]), "bjd_reference": float(h1["BJDREFI"]) + float(h1["BJDREFF"]),
                    "time_system": str(h1["TIMESYS"]), "rows_all": int(len(t)),
                    "rows_finite": int(np.sum(np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0))),
                    "rows_quality_nonzero": int(np.sum(q != 0)), "file_sha256": sha(path)}
    keep = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (q == 0)
    order = np.argsort(t[keep])
    return t[keep][order], f[keep][order], e[keep][order], metadata


def mask(t: np.ndarray, p: float, epoch: float, duration: float) -> np.ndarray:
    phase = (t - epoch + 0.5 * p) % p - 0.5 * p
    return np.abs(phase) <= 0.5 * duration


def sinusoid(t: np.ndarray, p: float, coefficients: list[float]) -> np.ndarray:
    a, b, c = coefficients
    theta = 2 * np.pi * t / p
    return a + b * np.sin(theta) + c * np.cos(theta)


def box(t: np.ndarray, model: dict) -> np.ndarray:
    return float(model["baseline"]) - float(model["depth"]) * mask(
        t, float(model["period_days"]), float(model["epoch_bkjd"]), float(model["duration_days"]))


def metrics(t: np.ndarray, y: np.ndarray, models: dict) -> dict:
    inside = mask(t, models["box"]["period_days"], models["box"]["epoch_bkjd"], models["box"]["duration_days"])
    predictions = {
        "constant": np.full_like(y, models["constant"]["baseline"]),
        "sinusoid": sinusoid(t, models["sinusoid"]["period_days"], models["sinusoid"]["coefficients"]),
        "box": box(t, models["box"]),
    }
    return {
        "n": int(len(y)), "predicted_transit_n": int(inside.sum()), "other_n": int((~inside).sum()),
        "observed_mean_transit": float(np.mean(y[inside])) if inside.any() else None,
        "observed_mean_other": float(np.mean(y[~inside])) if (~inside).any() else None,
        "rmse": {name: float(np.sqrt(np.mean((pred - y) ** 2))) for name, pred in predictions.items()},
        "rmse_predicted_transit": {name: float(np.sqrt(np.mean((pred[inside] - y[inside]) ** 2))) if inside.any() else None for name, pred in predictions.items()},
        "rmse_other": {name: float(np.sqrt(np.mean((pred[~inside] - y[~inside]) ** 2))) if (~inside).any() else None for name, pred in predictions.items()},
    }


def train(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    t, f, e, metadata = load("q1")
    split_time = float(t[0] + 0.7 * (t[-1] - t[0]))
    fit = t < split_time
    scale = float(np.median(f[fit]))
    y, err = f / scale, e / scale
    x, yy, ee = t[fit], y[fit], err[fit]

    frequency_grid = np.linspace(1 / 5, 1 / 1, 4000)
    ls = LombScargle(x, yy, dy=ee)
    ls_power = ls.power(frequency_grid)
    ls_frequency = float(frequency_grid[int(np.argmax(ls_power))])
    sinusoid_period = 1 / ls_frequency
    theta = 2 * np.pi * x / sinusoid_period
    design = np.stack([np.ones_like(x), np.sin(theta), np.cos(theta)], axis=1)
    coef = np.linalg.lstsq(design / ee[:, None], yy / ee, rcond=None)[0]

    bls = BoxLeastSquares(x, yy, dy=ee)
    period_grid = np.linspace(1, 5, 4000)
    coarse = bls.power(period_grid, DURATIONS, objective="likelihood")
    j = int(np.argmax(coarse.power))
    near = np.linspace(max(1.0, float(coarse.period[j]) - 0.004), min(5.0, float(coarse.period[j]) + 0.004), 2001)
    fine = bls.power(near, DURATIONS, objective="likelihood")
    k = int(np.argmax(fine.power))
    bp = float(fine.period[k]); bd = float(fine.duration[k]); epoch = float(fine.transit_time[k])
    within = mask(x, bp, epoch, bd)
    weights = 1 / ee**2
    baseline = float(np.average(yy[~within], weights=weights[~within]))
    low = float(np.average(yy[within], weights=weights[within]))
    models = {
        "constant": {"baseline": float(np.average(yy, weights=weights))},
        "sinusoid": {"period_days": sinusoid_period, "coefficients": [float(v) for v in coef], "train_search_power": float(ls_power.max())},
        "box": {"period_days": bp, "duration_days": bd, "epoch_bkjd": epoch, "baseline": baseline, "depth": baseline - low,
                "coarse_period_days": float(coarse.period[j]), "train_search_power": float(fine.power[k])},
    }
    report = {"scope": "Q1 fitting prefix and later Q1 validation, no Q2 loaded", "q1": metadata,
              "source": str(FILES["q1"]), "code_sha256": sha(Path(__file__)),
              "pretest_decision": "work/verification/C45_kepler_pretest_decision.md",
              "split_bkjd": split_time, "normalization_train_median_e_per_s": scale,
              "fit": metrics(x, yy, models), "validation": metrics(t[~fit], y[~fit], models), "models": models,
              "time_ranges_bkjd": {"fit": [float(x[0]), float(x[-1])], "validation": [float(t[~fit][0]), float(t[-1])]},
              "train_search": {"period_range_days": [1, 5], "coarse_points": 4000, "fine_points": 2001,
                               "durations_days": DURATIONS.tolist(), "ls_frequency_points": 4000}}
    (run_dir / "q1_train_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": models, "fit": report["fit"], "validation": report["validation"]}, indent=2))


def test(run_dir: Path) -> None:
    train_path = run_dir / "q1_train_validation.json"
    report = json.loads(train_path.read_text(encoding="utf-8"))
    if sha(Path(__file__)) != report["code_sha256"]:
        raise RuntimeError("Study code changed since Q1 decision")
    if sha(FILES["q1"]) != report["q1"]["file_sha256"]:
        raise RuntimeError("Q1 source changed")
    if (run_dir / "q2_first_test.json").exists():
        raise FileExistsError("Original Q2 test already exists")
    t, f, e, metadata = load("q2")
    cal = t < t[0] + 3.0
    if int(cal.sum()) < 50:
        raise ValueError("Too few Q2 calibration observations")
    scale = float(np.median(f[cal]))
    result = {"scope": "Q2 first out-of-quarter evaluation after 3-day no-label baseline calibration",
              "q1_training_report_sha256": sha(train_path), "q2": metadata,
              "calibration_days": 3.0, "calibration_n": int(cal.sum()), "scored_n": int((~cal).sum()),
              "calibration_median_e_per_s": scale,
              "scored_time_range_bkjd": [float(t[~cal][0]), float(t[-1])],
              "metrics": metrics(t[~cal], f[~cal] / scale, report["models"])}
    (run_dir / "q2_first_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "test"])
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    (train if args.stage == "train" else test)(args.run_dir)
