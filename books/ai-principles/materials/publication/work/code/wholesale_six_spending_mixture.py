"""在六项原消费的固定对数尺度上训练两成分混合并补一项缺失消费。"""

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

from wholesale_unsupervised import (
    WORK, FEATURES, fit_pca, load_data, split_rows, train_only_transform,
)


PRIOR_REPORT = WORK / "results/wholesale_unsupervised.json"
PRIOR_SHA256 = "6b13ae933d3241ab64c4bfb89d8435297ca1afaf12faf936cac0816c01df788c"
RESULTS = WORK / "results"
GROCERY_INDEX = FEATURES.index("Grocery")
SMALL = 1e-8


def load_six_dimensions():
    raw_report = PRIOR_REPORT.read_bytes()
    if hashlib.sha256(raw_report).hexdigest() != PRIOR_SHA256:
        raise ValueError("第八章结果版本变化，不能沿旧切分继续")
    prior = json.loads(raw_report)
    raw = load_data()  # 同时核 UCI 原 ZIP、CSV 的固定 SHA-256
    fit_ids, holdout_ids = split_rows(len(raw))
    if [int(i + 1) for i in fit_ids] != prior["fit_source_rows"]:
        raise ValueError("拟合行号与第八章不同")
    if [int(i + 1) for i in holdout_ids] != prior["holdout_source_rows"]:
        raise ValueError("留出行号与第八章不同")
    x_fit, x_holdout, log_mean, log_std = train_only_transform(
        raw[fit_ids], raw[holdout_ids]
    )
    _, _, directions = fit_pca(x_fit)
    if not np.allclose(log_mean, prior["train_only_log1p_mean"], atol=1e-12):
        raise ValueError("训练期对数均值与第八章不同")
    if not np.allclose(log_std, prior["train_only_log1p_std"], atol=1e-12):
        raise ValueError("训练期对数标准差与第八章不同")
    if not np.allclose(directions, prior["pca"]["directions_columns"], atol=1e-12):
        raise ValueError("画图所用 PCA 方向与第八章不同")
    centers = np.asarray(
        prior["two_centers"]["centers_in_standardized_log_space"], dtype=np.float64
    )
    weights = np.asarray(
        prior["two_centers"]["fit_group_sizes"], dtype=np.float64
    ) / len(x_fit)
    squared = np.sum((x_fit[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    variance = float(np.min(squared, axis=1).mean() / x_fit.shape[1])
    return (
        raw, x_fit, x_holdout, fit_ids, holdout_ids, log_mean, log_std,
        directions, weights, centers, variance,
    )


def log_normal(x, means, variance):
    if variance < SMALL or x.ndim != 2 or means.ndim != 2:
        raise ValueError("高斯密度的方差或数组维数不合法")
    dimension = x.shape[1]
    if means.shape[1] != dimension:
        raise ValueError("观测与成分均值维数不合")
    squared = np.sum((x[:, None, :] - means[None, :, :]) ** 2, axis=2)
    return -0.5 * dimension * math.log(2 * math.pi * variance) - squared / (2 * variance)


def logsumexp_each_row(values):
    high = values.max(axis=1, keepdims=True)
    return (high + np.log(np.exp(values - high).sum(axis=1, keepdims=True)))[:, 0]


def component_terms(x, weights, means, variance):
    if np.any(weights < SMALL) or not np.isclose(weights.sum(), 1, atol=1e-12):
        raise ValueError("混合权重不合法")
    return np.log(weights)[None, :] + log_normal(x, means, variance)


def log_density(x, weights, means, variance):
    return logsumexp_each_row(component_terms(x, weights, means, variance))


def posterior(x, weights, means, variance):
    terms = component_terms(x, weights, means, variance)
    totals = logsumexp_each_row(terms)
    result = np.exp(terms - totals[:, None])
    if not np.allclose(result.sum(axis=1), 1, atol=1e-12):
        raise ValueError("责任不满足逐行加一")
    return result


def bound_gap_probe(row, fit, source_row):
    """核对一行的对数似然下界差恰是 q 到当前后验的 KL。"""
    log_joint = component_terms(
        row[None, :], fit["weights"], fit["means"], fit["variance"]
    )
    log_observed = float(logsumexp_each_row(log_joint)[0])
    current_posterior = np.exp(log_joint[0] - log_observed)
    proposed_q = np.asarray([0.5, 0.5], dtype=np.float64)
    lower_bound = float(np.sum(proposed_q * (log_joint[0] - np.log(proposed_q))))
    kl_to_posterior = float(np.sum(proposed_q * np.log(proposed_q / current_posterior)))
    tight_bound = float(np.sum(
        current_posterior * (log_joint[0] - np.log(current_posterior))
    ))
    assert abs(log_observed - lower_bound - kl_to_posterior) < 1e-12
    assert abs(log_observed - tight_bound) < 1e-12
    return {
        "source_row": source_row,
        "proposed_q": proposed_q.tolist(),
        "current_posterior": current_posterior.tolist(),
        "observed_log_density": log_observed,
        "proposed_lower_bound": lower_bound,
        "gap_to_bound": log_observed - lower_bound,
        "kl_q_to_current_posterior": kl_to_posterior,
        "tight_gap_with_posterior": log_observed - tight_bound,
    }


def one_component_fit(x):
    mean = x.mean(axis=0)
    variance = float(np.sum((x - mean) ** 2) / (len(x) * x.shape[1]))
    if variance < SMALL:
        raise ValueError("单成分共同方差过小")
    return mean, variance


def fit_two_component_em(x, weights0, means0, variance0, pc1):
    weights = weights0.copy()
    means = means0.copy()
    variance = float(variance0)
    trace = [float(log_density(x, weights, means, variance).mean())]
    status = "max_200_iterations"
    for iteration in range(1, 201):
        r = posterior(x, weights, means, variance)
        n_k = r.sum(axis=0)
        if np.any(n_k < SMALL):
            raise ValueError(f"第 {iteration} 轮有成分责任总量过小")
        next_weights = n_k / len(x)
        next_means = (r.T @ x) / n_k[:, None]
        squared = np.sum((x[:, None, :] - next_means[None, :, :]) ** 2, axis=2)
        next_variance = float(
            np.sum(r * squared) / (len(x) * x.shape[1])
        )
        if next_variance < SMALL:
            raise ValueError(f"第 {iteration} 轮共同方差过小")
        next_ll = float(
            log_density(x, next_weights, next_means, next_variance).mean()
        )
        gain = next_ll - trace[-1]
        if gain < -1e-9:
            raise ValueError(f"第 {iteration} 轮观察似然反而下降 {gain}")
        trace.append(next_ll)
        weights, means, variance = next_weights, next_means, next_variance
        if gain < 1e-8:
            status = "gain_below_1e-8"
            break
    order = np.argsort(means @ pc1)
    return {
        "weights": weights[order],
        "means": means[order],
        "variance": variance,
        "trace": trace,
        "iterations": iteration,
        "status": status,
    }


def masked_grocery_example(
    raw, x_holdout, holdout_ids, log_mean, log_std, fit
):
    if int(holdout_ids[0] + 1) != 152:
        raise ValueError("预定的第一条留出行不再是原文件第 152 行")
    missing = GROCERY_INDEX
    keep = [j for j in range(len(FEATURES)) if j != missing]
    observed_five = x_holdout[0:1, keep]
    responsibility = posterior(
        observed_five,
        fit["weights"],
        fit["means"][:, keep],
        fit["variance"],
    )[0]
    full_responsibility = posterior(
        x_holdout[0:1], fit["weights"], fit["means"], fit["variance"]
    )[0]
    # 若标准化对数值 X_j|Z=k ~ N(m_k, variance)，
    # 则 log(1+原消费) 的均值与方差作如下仿射变换。
    mean_log_by_component = (
        log_mean[missing] + log_std[missing] * fit["means"][:, missing]
    )
    variance_log = float(log_std[missing] ** 2 * fit["variance"])
    expected_raw_by_component = (
        np.exp(mean_log_by_component + 0.5 * variance_log) - 1
    )
    expected_raw = float(responsibility @ expected_raw_by_component)
    return {
        "source_row": 152,
        "observed_columns": [FEATURES[j] for j in keep],
        "observed_raw_values": {
            FEATURES[j]: float(raw[holdout_ids[0], j]) for j in keep
        },
        "masked_column": "Grocery",
        "posterior_given_only_five_observed_columns": responsibility.tolist(),
        "model_expected_grocery_monetary_units": expected_raw,
        "model_expected_grocery_by_component": expected_raw_by_component.tolist(),
        "actual_grocery_revealed_only_for_check": float(
            raw[holdout_ids[0], missing]
        ),
        "posterior_after_revealing_grocery_for_description_only":
            full_responsibility.tolist(),
        "scope": "One already-known archived holdout row, hidden Grocery before inference. Model has never fit this row, but this archive was examined in the prior chapter; result is exploratory.",
    }


def softmax(logits):
    high = float(np.max(logits))
    weights = np.exp(logits - high)
    return weights / weights.sum()


def factor_table():
    f1 = np.array([1.0, 2.0])
    f12 = np.array([[4.0, 1.0], [1.0, 4.0]])
    f2 = np.array([1.0, 1.0])
    unnormalized = f1[:, None] * f12 * f2[None, :]
    total = float(unnormalized.sum())
    exact = unnormalized / total
    q1, q2 = np.array([0.5, 0.5]), np.array([0.5, 0.5])
    kl_trace = []
    for iteration in range(1, 101):
        old1, old2 = q1.copy(), q2.copy()
        q1 = softmax(np.log(f1) + (np.log(f12) * q2[None, :]).sum(axis=1))
        q2 = softmax(np.log(f2) + (np.log(f12) * q1[:, None]).sum(axis=0))
        q_joint = q1[:, None] * q2[None, :]
        kl = float(np.sum(q_joint * np.log(q_joint / exact)))
        kl_trace.append(kl)
        if len(kl_trace) > 1 and kl_trace[-1] > kl_trace[-2] + 1e-12:
            raise ValueError("均值场坐标更新使 KL 升高")
        if max(float(np.max(np.abs(q1 - old1))),
               float(np.max(np.abs(q2 - old2)))) < 1e-10:
            break
    else:
        raise RuntimeError("两隐藏位 100 轮内未稳定")
    return {
        "factors": {
            "f1": f1.tolist(), "f12": f12.tolist(), "f2": f2.tolist()
        },
        "normalizer": total,
        "exact_joint_rows_z1_columns_z2": exact.tolist(),
        "exact_marginal_z1": exact.sum(axis=1).tolist(),
        "exact_marginal_z2": exact.sum(axis=0).tolist(),
        "mean_field_q1": q1.tolist(),
        "mean_field_q2": q2.tolist(),
        "mean_field_joint": q_joint.tolist(),
        "kl_q_to_exact": kl,
        "kl_trace": kl_trace,
        "iterations": iteration,
        "scope": "Author-made four-state factor table. Exact sum is available; factorized q is used to display lost dependence, not to model customers.",
    }


def draw_responsibility_figure(x_fit, directions, fit):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    projected = x_fit @ directions[:, :2]
    colors = posterior(
        x_fit, fit["weights"], fit["means"], fit["variance"]
    )[:, 1]
    projected_means = fit["means"] @ directions[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    scatter = axes[0].scatter(
        projected[:, 0], projected[:, 1], c=colors, cmap="viridis",
        s=21, vmin=0, vmax=1, alpha=0.85
    )
    axes[0].scatter(
        projected_means[:, 0], projected_means[:, 1],
        marker="X", s=160, c="white", edgecolors="black", linewidths=1
    )
    axes[0].set_xlabel("第一主方向坐标（仅展示）")
    axes[0].set_ylabel("第二主方向坐标（仅展示）")
    axes[0].set_title("后验用六项算，图只画两轴")
    axes[0].grid(alpha=0.15)
    fig.colorbar(scatter, ax=axes[0], label="模型 P(Z=1 | 六项全见)")
    axes[1].hist(
        colors, bins=np.linspace(0, 1, 21), color="#517f93", edgecolor="white"
    )
    axes[1].set_xlabel("模型 P(Z=1 | 六项全见)")
    axes[1].set_ylabel("拟合行数")
    axes[1].set_title("各行对成分 1 的责任")
    axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle("六维标准化对数消费的两成分模型",
                 fontsize=12)
    fig.tight_layout()
    target = WORK / "figures/wholesale_six_spending_responsibilities.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def draw_structure_figure():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    left = axes[0]
    left.set_title("选定的混合模型分解", fontsize=12)
    left.add_patch(Circle((0.5, 0.8), 0.075, facecolor="#e4d9b9",
                          edgecolor="#30343b"))
    left.text(0.5, 0.8, "Z", ha="center", va="center", fontsize=13)
    names = ["生鲜", "乳品", "杂货", "冷冻", "清洁纸品", "熟食"]
    for j, (x, name) in enumerate(zip(np.linspace(0.08, 0.92, 6), names), start=1):
        left.annotate("", xy=(x, 0.43), xytext=(0.5, 0.72),
                      arrowprops={"arrowstyle": "->", "color": "#59636e",
                                  "lw": 1.2})
        left.add_patch(Circle((x, 0.34), 0.053, facecolor="#dceaf6",
                              edgecolor="#30343b"))
        left.text(x, 0.34, f"X{j}", ha="center", va="center", fontsize=10)
        left.text(x, 0.2, name, ha="center", va="center", fontsize=8)
    left.text(0.5, 0.05, "箭头表示本模型的条件密度乘积，不表示干预。",
              ha="center", fontsize=8)

    right = axes[1]
    right.set_title("作者构造的双隐藏位因子表", fontsize=12)
    chain = [
        (0.08, "f1", "factor"), (0.28, "Z1", "variable"),
        (0.5, "f12", "factor"), (0.72, "Z2", "variable"),
        (0.92, "f2", "factor"),
    ]
    for left_x, right_x in zip([0.12, 0.33, 0.55, 0.77],
                               [0.23, 0.45, 0.67, 0.87]):
        right.plot([left_x, right_x], [0.55, 0.55],
                   color="#59636e", linewidth=1.5)
    for x, label, kind in chain:
        if kind == "variable":
            right.add_patch(Circle((x, 0.55), 0.053, facecolor="#e4d9b9",
                                   edgecolor="#30343b", zorder=2))
        else:
            right.add_patch(FancyBboxPatch(
                (x - 0.05, 0.5), 0.1, 0.1,
                boxstyle="round,pad=0.01", facecolor="#dceaf6",
                edgecolor="#30343b", zorder=2
            ))
        right.text(x, 0.55, label, ha="center", va="center",
                   fontsize=10, zorder=3)
    right.text(0.08, 0.31, "已见证据", ha="center", fontsize=9)
    right.text(0.5, 0.31, "同值偏好", ha="center", fontsize=9)
    right.text(0.5, 0.05, "先局部相乘，再对隐藏状态求和。",
               ha="center", fontsize=8)
    fig.tight_layout()
    target = WORK / "figures/wholesale_mixture_factorization.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def main():
    (
        raw, x_fit, x_holdout, fit_ids, holdout_ids, log_mean, log_std,
        directions, initial_weights, initial_centers, initial_variance,
    ) = load_six_dimensions()
    one_mean, one_variance = one_component_fit(x_fit)
    fitted = fit_two_component_em(
        x_fit, initial_weights, initial_centers, initial_variance,
        directions[:, 0],
    )
    one_fit = float(log_normal(x_fit, one_mean[None, :], one_variance).mean())
    one_reused = float(log_normal(
        x_holdout, one_mean[None, :], one_variance
    ).mean())
    two_fit = float(log_density(
        x_fit, fitted["weights"], fitted["means"], fitted["variance"]
    ).mean())
    two_reused = float(log_density(
        x_holdout, fitted["weights"], fitted["means"], fitted["variance"]
    ).mean())
    if not np.isclose(two_fit, fitted["trace"][-1], atol=1e-12):
        raise ValueError("最后参数与观察似然轨迹不一致")
    missing = masked_grocery_example(
        raw, x_holdout, holdout_ids, log_mean, log_std, fitted
    )
    bound_probe = bound_gap_probe(x_fit[0], fitted, int(fit_ids[0] + 1))
    toy = factor_table()
    RESULTS.mkdir(exist_ok=True)
    report = {
        "protocol": "work/verification/C09_mixture_six_spending_protocol.md",
        "source": "work/data/uci_wholesale_SOURCE.md",
        "prior_chapter_result": "work/results/wholesale_unsupervised.json",
        "prior_chapter_result_sha256": PRIOR_SHA256,
        "supersedes_for_raw_missing_question": "work/verification/C09_mixture_protocol.md",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "fitted_columns": FEATURES,
        "excluded_columns": ["Channel", "Region"],
        "fit_rows": len(x_fit),
        "reused_archive_holdout_rows": len(x_holdout),
        "single_component": {
            "mean_in_standardized_log_space": one_mean.tolist(),
            "shared_variance": one_variance,
            "fit_mean_log_density": one_fit,
            "reused_holdout_mean_log_density": one_reused,
        },
        "two_component": {
            "initial_weights": initial_weights.tolist(),
            "initial_centers_in_six_dimensions": initial_centers.tolist(),
            "initial_variance": initial_variance,
            "weights": fitted["weights"].tolist(),
            "centers_in_standardized_log_space": fitted["means"].tolist(),
            "shared_variance": fitted["variance"],
            "iterations": fitted["iterations"],
            "stop_reason": fitted["status"],
            "mean_log_likelihood_trace": fitted["trace"],
            "fit_mean_log_density": two_fit,
            "reused_holdout_mean_log_density": two_reused,
        },
        "masked_grocery_example": missing,
        "one_row_em_lower_bound_probe": bound_probe,
        "author_made_factor_table": toy,
        "figure": "work/figures/wholesale_six_spending_responsibilities.png",
        "factorization_figure": "work/figures/wholesale_mixture_factorization.png",
        "scope": "Two-component shared-isotropic-variance Gaussian mixture on all six train-standardized log spending columns, 352 fit rows. The 88 archive holdout rows were already examined in prior work and only support exploratory checks. Components are not known customer types.",
    }
    result_file = RESULTS / "wholesale_six_spending_mixture.json"
    result_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (RESULTS / "wholesale_six_spending_responsibilities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "source_row", "role", "model_responsibility_0",
            "model_responsibility_1",
        ])
        for role, ids, observations in (
            ("fit", fit_ids, x_fit),
            ("reused_holdout", holdout_ids, x_holdout),
        ):
            probs = posterior(
                observations, fitted["weights"], fitted["means"], fitted["variance"]
            )
            for source_id, responsibility in zip(ids, probs):
                writer.writerow([
                    int(source_id + 1), role,
                    float(responsibility[0]), float(responsibility[1]),
                ])
    draw_responsibility_figure(x_fit, directions, fitted)
    draw_structure_figure()
    print("EM 轮数/停止原因:", fitted["iterations"], fitted["status"])
    print("单成分六维拟合/已见留出平均对数密度:",
          round(one_fit, 5), round(one_reused, 5))
    print("双成分六维拟合/已见留出平均对数密度:",
          round(two_fit, 5), round(two_reused, 5))
    print("双成分权重/共同方差:", fitted["weights"], fitted["variance"])
    print("第 152 行遮住 Grocery 的模型预测:", missing)
    print("一行的下界差与 KL:", bound_probe["gap_to_bound"],
          bound_probe["kl_q_to_current_posterior"])
    print("小表精确联合/因子近似:", toy["exact_joint_rows_z1_columns_z2"],
          toy["mean_field_joint"], "KL", toy["kl_q_to_exact"])


if __name__ == "__main__":
    main()
