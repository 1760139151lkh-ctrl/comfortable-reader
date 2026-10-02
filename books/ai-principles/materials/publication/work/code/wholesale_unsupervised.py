"""C09 前半主题：仅用六项消费额核对 PCA、两个代表点和尺度选择。"""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import random
import statistics
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORK = Path(__file__).resolve().parents[1]
ARCHIVE = WORK / "data/uci_wholesale_original.zip"
RAW_DIR = WORK / "data/uci_wholesale"
CSV_NAME = "Wholesale customers data.csv"
RAW_FILE = RAW_DIR / CSV_NAME
URL = "https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip"
ZIP_SHA256 = "647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8"
CSV_SHA256 = "c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef"
FEATURES = ("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")
HEADER = ("Channel", "Region", *FEATURES)
SPLIT_SEED = 20260923
INIT_SEEDS = (11, 22, 33, 44, 55)
N_FIT = 352
N_HOLDOUT = 88


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_data():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        ARCHIVE.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
    archive_bytes = ARCHIVE.read_bytes()
    if digest(archive_bytes) != ZIP_SHA256:
        raise ValueError("UCI 原包字节身份不符")
    with ZipFile(ARCHIVE) as source:
        if source.namelist() != [CSV_NAME]:
            raise ValueError("原包成员名称或数量有变")
        raw = source.read(CSV_NAME)
    if digest(raw) != CSV_SHA256:
        raise ValueError("CSV 原字节哈希不符")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and RAW_FILE.read_bytes() != raw:
        raise ValueError("已有 CSV 与原包不同，不覆盖")
    if not RAW_FILE.exists():
        RAW_FILE.write_bytes(raw)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != HEADER:
        raise ValueError("原 CSV 列名或次序有变")
    records = list(reader)
    if len(records) != 440:
        raise ValueError("记录行数不等于 440")
    measurements = []
    for index, record in enumerate(records, start=1):
        if None in record.values():
            raise ValueError(f"第 {index} 行存在空值")
        values = [float(record[name]) for name in FEATURES]
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError(f"第 {index} 行消费额不是非负有限数")
        measurements.append(values)
    return np.asarray(measurements, dtype=np.float64)


def split_rows(n):
    indices = list(range(n))
    random.Random(SPLIT_SEED).shuffle(indices)
    fit = np.asarray(indices[:N_FIT], dtype=np.int64)
    holdout = np.asarray(indices[N_FIT:], dtype=np.int64)
    if len(fit) != N_FIT or len(holdout) != N_HOLDOUT:
        raise ValueError("预定的 352/88 切分失败")
    if len(set(fit.tolist()) & set(holdout.tolist())):
        raise ValueError("拟合与留出行有交叠")
    return fit, holdout


def train_only_transform(raw_fit, raw_holdout, mode="log1p"):
    if mode == "log1p":
        fit_values, holdout_values = np.log1p(raw_fit), np.log1p(raw_holdout)
    elif mode == "raw":
        fit_values, holdout_values = raw_fit, raw_holdout
    else:
        raise ValueError("未知的尺度选择")
    mean = fit_values.mean(axis=0)
    std = fit_values.std(axis=0)
    if np.any(std == 0):
        raise ValueError("拟合段有零标准差的商品列")
    return (fit_values - mean) / std, (holdout_values - mean) / std, mean, std


def fit_pca(z):
    covariance = z.T @ z / len(z)
    eigenvalues, directions = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    directions = directions[:, order]
    for j in range(directions.shape[1]):
        largest = int(np.argmax(np.abs(directions[:, j])))
        if directions[largest, j] < 0:
            directions[:, j] *= -1
    if not np.allclose(directions.T @ directions, np.eye(6), atol=1e-10):
        raise ValueError("主方向未能组成正交单位向量组")
    return covariance, eigenvalues, directions


def reconstruction_error(z, directions, rank):
    basis = directions[:, :rank]
    reconstructed = (z @ basis) @ basis.T
    return float(np.mean(np.sum((z - reconstructed) ** 2, axis=1)))


def squared_distances(z, centers):
    return np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def fit_two_centers(z, fit_ids, seed):
    selected = np.random.default_rng(seed).choice(len(z), size=2, replace=False)
    centers = z[selected].copy()
    previous = None
    trace = []
    for iteration in range(1, 101):
        groups = np.argmin(squared_distances(z, centers), axis=1)
        if any(int(np.count_nonzero(groups == j)) == 0 for j in range(2)):
            raise ValueError(f"种子 {seed} 的第 {iteration} 轮出现空组")
        new_centers = np.stack([z[groups == j].mean(axis=0) for j in range(2)])
        trace.append(float(np.mean(np.min(squared_distances(z, new_centers), axis=1))))
        if previous is not None and np.array_equal(groups, previous):
            centers = new_centers
            break
        centers = new_centers
        previous = groups
    else:
        raise RuntimeError(f"种子 {seed} 超过 100 轮仍未稳定")
    final_groups = np.argmin(squared_distances(z, centers), axis=1)
    if not np.array_equal(final_groups, groups):
        raise ValueError("停止后分组又改变")
    if any(trace[i + 1] > trace[i] + 1e-10 for i in range(len(trace) - 1)):
        raise ValueError("批量交替的拟合目标意外升高")
    objective = float(np.mean(np.min(squared_distances(z, centers), axis=1)))
    return {
        "seed": seed,
        "initial_source_rows": [int(fit_ids[i] + 1) for i in selected],
        "centers": centers,
        "groups": final_groups,
        "fit_objective": objective,
        "iterations": iteration,
        "objective_trace": trace,
    }


def draw_figure(z_fit, centers, groups, eigenvalues, directions, mode="log1p"):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    scores = z_fit @ directions[:, :2]
    center_scores = centers @ directions[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ("#24527a", "#a54d13")
    for group in (0, 1):
        mask = groups == group
        axes[0].scatter(scores[mask, 0], scores[mask, 1], s=20, alpha=0.75,
                        c=colors[group], label=f"算法组 {group}")
        axes[0].scatter(center_scores[group, 0], center_scores[group, 1],
                        s=150, marker="X", c=colors[group], edgecolors="black",
                        linewidths=0.8)
    axes[0].set_xlabel("第一主方向坐标")
    axes[0].set_ylabel("第二主方向坐标")
    axes[0].set_title("六项消费投到两个方向")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.15)
    fractions = eigenvalues / eigenvalues.sum()
    axes[1].bar(range(1, 7), fractions, color="#517f93")
    axes[1].set_xticks(range(1, 7))
    axes[1].set_xlabel("主方向序号")
    axes[1].set_ylabel("拟合段方差份额")
    axes[1].set_title("二维图保留了多少变化")
    axes[1].set_ylim(0, max(0.5, float(fractions.max() + 0.05)))
    axes[1].grid(axis="y", alpha=0.15)
    scale_name = "对数后标准化" if mode == "log1p" else "原消费额标准化"
    fig.suptitle(f"无客户类型标签：{scale_name}", fontsize=13)
    fig.tight_layout()
    suffix = "" if mode == "log1p" else "_raw"
    target = WORK / f"figures/wholesale_pca_kmeans{suffix}.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description="比较对数尺度与原数尺度上的无标签结构")
    parser.add_argument("--transform", choices=("log1p", "raw"), default="log1p")
    args = parser.parse_args()
    mode = args.transform
    raw = load_data()
    fit_ids, holdout_ids = split_rows(len(raw))
    raw_fit, raw_holdout = raw[fit_ids], raw[holdout_ids]
    z_fit, z_holdout, fit_mean, fit_std = train_only_transform(raw_fit, raw_holdout, mode)
    covariance, eigenvalues, directions = fit_pca(z_fit)
    variance_fractions = eigenvalues / eigenvalues.sum()
    pca = {
        "covariance": covariance.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "variance_fractions": variance_fractions.tolist(),
        "first_two_fraction": float(variance_fractions[:2].sum()),
        "directions_columns": directions.tolist(),
        "rank1_fit_error": reconstruction_error(z_fit, directions, 1),
        "rank1_holdout_error": reconstruction_error(z_holdout, directions, 1),
        "rank2_fit_error": reconstruction_error(z_fit, directions, 2),
        "rank2_holdout_error": reconstruction_error(z_holdout, directions, 2),
    }
    if not np.isclose(pca["rank2_fit_error"], eigenvalues[2:].sum(), atol=1e-10):
        raise ValueError("PCA 重建误差与剩余特征值不一致")

    trials = [fit_two_centers(z_fit, fit_ids, seed) for seed in INIT_SEEDS]
    chosen = min(trials, key=lambda item: (item["fit_objective"], INIT_SEEDS.index(item["seed"])))
    order = np.argsort(chosen["centers"] @ directions[:, 0])
    centers = chosen["centers"][order]
    fit_groups = np.argmin(squared_distances(z_fit, centers), axis=1)
    holdout_groups = np.argmin(squared_distances(z_holdout, centers), axis=1)
    fit_distortion = float(np.mean(np.min(squared_distances(z_fit, centers), axis=1)))
    holdout_distortion = float(np.mean(np.min(squared_distances(z_holdout, centers), axis=1)))
    if not np.isclose(fit_distortion, chosen["fit_objective"], atol=1e-10):
        raise ValueError("组号重排改变了平方距离目标")

    medians = {
        str(group): {
            name: float(statistics.median(raw_fit[fit_groups == group, j]))
            for j, name in enumerate(FEATURES)
        }
        for group in (0, 1)
    }
    result = {
        "source": "work/data/uci_wholesale_SOURCE.md",
        "protocol": "work/verification/C09_wholesale_protocol.md",
        "zip_sha256": ZIP_SHA256,
        "csv_sha256": CSV_SHA256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "feature_names": FEATURES,
        "excluded_columns": ["Channel", "Region"],
        "split_seed": SPLIT_SEED,
        "fit_source_rows": [int(i + 1) for i in fit_ids],
        "holdout_source_rows": [int(i + 1) for i in holdout_ids],
        "transform": mode,
        f"train_only_{mode}_mean": fit_mean.tolist(),
        f"train_only_{mode}_std": fit_std.tolist(),
        "pca": pca,
        "two_centers": {
            "seeds_predeclared": INIT_SEEDS,
            "trials": [{
                key: value for key, value in trial.items()
                if key not in ("centers", "groups")
            } for trial in trials],
            "chosen_seed_by_fit_objective": chosen["seed"],
            "centers_in_standardized_log_space": centers.tolist(),
            "fit_group_sizes": [int(np.count_nonzero(fit_groups == g)) for g in (0, 1)],
            "holdout_group_sizes": [int(np.count_nonzero(holdout_groups == g)) for g in (0, 1)],
            "fit_average_squared_distance": fit_distortion,
            "holdout_average_squared_distance": holdout_distortion,
            "raw_spending_medians_by_algorithm_group": medians,
        },
        "figure": "work/figures/wholesale_pca_kmeans" + ("" if mode == "log1p" else "_raw") + ".png",
        "scope": "440 archived rows of annual wholesale spending; six numeric columns only. PCA and two-center partition fitted on 352 rows after train-only " + mode + " standardization, checked on 88 held-out archive rows. No customer-type ground truth or future-year claim.",
    }
    suffix = "" if mode == "log1p" else "_raw"
    out = WORK / f"results/wholesale_unsupervised{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out.parent / f"wholesale_group_assignments{suffix}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["source_row", "role", "algorithm_group", "pc1", "pc2"])
        for role, ids, z, groups in (
            ("fit", fit_ids, z_fit, fit_groups),
            ("holdout", holdout_ids, z_holdout, holdout_groups),
        ):
            scores = z @ directions[:, :2]
            for row_id, group, score in zip(ids, groups, scores):
                writer.writerow([int(row_id + 1), role, int(group),
                                 float(score[0]), float(score[1])])
    draw_figure(z_fit, centers, fit_groups, eigenvalues, directions, mode)
    print("拟合/留出行数:", len(fit_ids), len(holdout_ids))
    print("前两主方向解释拟合段方差份额:", round(pca["first_two_fraction"], 4))
    print("PCA rank1 训练/留出重建误差:", round(pca["rank1_fit_error"], 4),
          round(pca["rank1_holdout_error"], 4))
    print("PCA rank2 训练/留出重建误差:", round(pca["rank2_fit_error"], 4),
          round(pca["rank2_holdout_error"], 4))
    print("五次两中心拟合目标:", [round(t["fit_objective"], 4) for t in trials])
    print("选中种子:", chosen["seed"])
    print("算法分组拟合/留出人数:",
          result["two_centers"]["fit_group_sizes"],
          result["two_centers"]["holdout_group_sizes"])
    print("两中心拟合/留出平均平方距离:",
          round(fit_distortion, 4), round(holdout_distortion, 4))


if __name__ == "__main__":
    main()
