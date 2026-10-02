"""A two-state linear filter on the same simulated Foggia road trajectory.

No real sensor was used. Gaussian noise and the constant-lateral-rate model are
author choices; the filter is an estimation demonstration, not an AV sensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from c42_bicycle_control import DT, RUNS, Car
from c42_commonroad_scene import load_scene

WORK = Path(__file__).resolve().parents[1]
TRACE = WORK / "runs/c42_control_verified/trace.json"
ROAD_REPORT = WORK / "runs/c42_control_verified/report.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lateral_error(car, scene):
    s = scene.route.project(Point(car.x, car.y))
    near = scene.route.interpolate(s)
    before = scene.route.interpolate(max(0.0, s - 0.4))
    after = scene.route.interpolate(min(scene.route.length, s + 0.4))
    angle = math.atan2(after.y - before.y, after.x - before.x)
    return (car.x - near.x) * -math.sin(angle) + (car.y - near.y) * math.cos(angle)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main(out_dir: Path, seed: int, sigma: float):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists() or not (0.05 <= sigma <= 2):
        raise ValueError("choose new work/runs directory and sensor sigma in [0.05,2]")
    scene = load_scene()
    source = json.loads(ROAD_REPORT.read_text(encoding="utf-8"))
    if source["source_sha256"] != "2563a7dd4eedb60ef460d37afe4c0b718510092289f81341eb967bd3f741b8b4":
        raise RuntimeError("road source differs")
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    cars = [Car(**record) for record in trace["closed_bias"]["states"]]
    true = np.array([lateral_error(car, scene) for car in cars], dtype=float)
    rng = np.random.default_rng(seed)
    observed = true + rng.normal(0.0, sigma, size=true.size)
    true_rate = np.gradient(true, DT)
    raw_rate = np.gradient(observed, DT)
    A = np.array([[1.0, DT], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q = np.diag([0.0025, 0.03])
    R = sigma**2
    # Prior fixed before observing this trajectory; the first measurement is used once.
    estimate = np.array([0.0, 0.0])
    covariance = np.diag([1.0, 4.0])
    filtered = []
    gains = []
    for measurement in observed:
        predicted = A @ estimate
        pred_cov = A @ covariance @ A.T + Q
        innovation = measurement - (C @ predicted).item()
        denom = (C @ pred_cov @ C.T).item() + R
        gain = (pred_cov @ C.T)[:, 0] / denom
        estimate = predicted + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, C[0])) @ pred_cov
        filtered.append(estimate.copy())
        gains.append(gain.copy())
    filtered = np.stack(filtered)
    gains = np.stack(gains)
    out.mkdir(parents=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    seconds = np.arange(len(true)) * DT
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(seconds, true, label="仿真真横向偏差", color="black")
    axes[0].scatter(seconds, observed, label="作者加噪位置观测", s=7, alpha=0.35, color="#d67534")
    axes[0].plot(seconds, filtered[:, 0], label="二状态滤波估计", color="#177a7b")
    axes[0].set_ylabel("偏差（米）")
    axes[0].legend(loc="upper right")
    axes[1].plot(seconds, true_rate, label="仿真真偏差变化率", color="black")
    axes[1].plot(seconds, raw_rate, label="直接对带噪位置求差", color="#d67534", alpha=0.35)
    axes[1].plot(seconds, filtered[:, 1], label="滤波的变化率状态", color="#177a7b")
    axes[1].set_xlabel("仿真时间（秒）")
    axes[1].set_ylabel("偏差变化率（米/秒）")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.17)
    fig.tight_layout()
    fig.savefig(out / "lateral_filter.png", dpi=150)
    plt.close(fig)
    report = {"road_report_sha256": sha(ROAD_REPORT), "road_trace_sha256": sha(TRACE),
              "vehicle_branch": "closed_bias", "samples": len(true), "dt_seconds": DT,
              "author_measurement_noise_sigma_m": sigma, "random_seed": seed,
              "model": {"A": A.tolist(), "C": C.tolist(), "Q": Q.tolist(), "R": R},
              "rmse_lateral_raw_m": rmse(observed, true),
              "rmse_lateral_filtered_m": rmse(filtered[:, 0], true),
              "rmse_rate_raw_m_s": rmse(raw_rate, true_rate),
              "rmse_rate_filtered_m_s": rmse(filtered[:, 1], true_rate),
              "first_gain": gains[0].tolist(), "last_gain": gains[-1].tolist(),
              "figure": "lateral_filter.png",
              "boundary": "synthetic noisy lateral position from this book's vehicle simulation, not real GPS/IMU; constant lateral rate is approximate on a curve; filtered estimate is not fed back into controller"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rmse_lateral_raw_m", "rmse_lateral_filtered_m",
                                           "rmse_rate_raw_m_s", "rmse_rate_filtered_m_s")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--sigma", type=float, default=0.4)
    args = parser.parse_args()
    main(args.out_dir, args.seed, args.sigma)
