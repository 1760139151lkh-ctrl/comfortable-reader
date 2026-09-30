"""同一黑箱评价预算下比较四种选点办法。运行：python work/code/black_box_budget.py"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "black_box_budget.json"
FIGURE = ROOT / "figures" / "black_box_budget.png"
METHODS = ("random", "local", "population", "gp_ei")
SEEDS = tuple(range(2026092401, 2026092413))
INITIAL = 6
BUDGET = 48
SUCCESS_VALUE = 0.5


def branin_hidden_formula(unit_points: np.ndarray) -> np.ndarray:
    """仅评价器与最后的作者作图使用；四个搜索器不可读取此函数。"""
    p = np.asarray(unit_points, dtype=float)
    x = -5.0 + 15.0 * p[..., 0]
    y = 15.0 * p[..., 1]
    return (
        (y - 5.1 * x**2 / (4 * np.pi**2) + 5 * x / np.pi - 6) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x)
        + 10
    )


@dataclass
class OracleLog:
    method: str
    seed: int
    points: list[list[float]]
    values: list[float]
    best_so_far: list[float]

    def __init__(self, method: str, seed: int) -> None:
        self.method, self.seed = method, seed
        self.points, self.values, self.best_so_far = [], [], []

    def evaluate(self, point: np.ndarray) -> float:
        if len(self.values) >= BUDGET:
            raise RuntimeError("评价次数已用完")
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("候选须为两个有限数")
        if np.any(point < 0) or np.any(point > 1):
            raise ValueError("候选越出了约定的搜索域")
        value = float(branin_hidden_formula(point))
        self.points.append(point.tolist())
        self.values.append(value)
        self.best_so_far.append(min(self.best_so_far[-1], value) if self.best_so_far else value)
        return value

    def best_point(self) -> np.ndarray:
        return np.asarray(self.points[int(np.argmin(self.values))], dtype=float)


def use_initial(log: OracleLog, initial: np.ndarray) -> None:
    for point in initial:
        log.evaluate(point)


def random_search(log: OracleLog, rng: np.random.Generator) -> None:
    while len(log.values) < BUDGET:
        log.evaluate(rng.uniform(size=2))


def local_search(log: OracleLog, rng: np.random.Generator) -> None:
    """一步一评；失败若干次后用一次预算另选起点。"""
    centre = log.best_point()
    centre_value = min(log.values)
    sigma = 0.18
    failures = 0
    while len(log.values) < BUDGET:
        if failures >= 8:
            point = rng.uniform(size=2)
            value = log.evaluate(point)
            centre, centre_value = point, value
            failures = 0
            sigma = 0.18
            continue
        else:
            point = np.clip(centre + rng.normal(scale=sigma, size=2), 0, 1)
        value = log.evaluate(point)
        if value < centre_value:
            centre, centre_value = point, value
            failures = 0
            sigma = min(0.3, sigma * 1.05)
        else:
            failures += 1
            sigma = max(0.015, sigma * 0.8)


def population_search(log: OracleLog, rng: np.random.Generator) -> None:
    """教学版 (mu+lambda) 实数群体：选择、重组、变异和移民。"""
    population = [(np.asarray(p), v) for p, v in zip(log.points, log.values)]
    generation = 0
    while len(log.values) < BUDGET:
        population.sort(key=lambda item: item[1])
        parents = population[:3]
        children = []
        sigma = max(0.04, 0.20 * (0.9**generation))
        for child_index in range(min(6, BUDGET - len(log.values))):
            if child_index == 5:
                point = rng.uniform(size=2)  # 防止整个群体只在一个局部范围繁殖
            else:
                left, right = rng.integers(0, len(parents), size=2)
                base = (parents[left][0] + parents[right][0]) / 2
                point = np.clip(base + rng.normal(scale=sigma, size=2), 0, 1)
            value = log.evaluate(point)
            children.append((point, value))
        population = sorted(population + children, key=lambda item: item[1])[:6]
        generation += 1


def kernel(a: np.ndarray, b: np.ndarray, length: float = 0.22) -> np.ndarray:
    scaled = (a[:, None, :] - b[None, :, :]) / length
    return np.exp(-0.5 * np.sum(scaled * scaled, axis=2))


def gp_predict(points: np.ndarray, values: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """固定尺度 RBF 核；输出是标准化评价值的预测均值和标准差。"""
    y_mean = float(np.mean(values))
    y_scale = max(float(np.std(values)), 1e-8)
    standardized = (values - y_mean) / y_scale
    gram = kernel(points, points) + np.eye(len(points)) * 1e-7
    cross = kernel(candidates, points)
    lower = np.linalg.cholesky(gram)
    alpha = solve_triangular(lower.T, solve_triangular(lower, standardized, lower=True))
    mean = cross @ alpha
    triangular = solve_triangular(lower, cross.T, lower=True)
    variance = np.maximum(0.0, 1.0 - np.sum(triangular * triangular, axis=0))
    return mean, np.sqrt(variance)


def expected_improvement(best: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """若 Y~N(mean,std²)，计算 E[max(best-Y,0)]。"""
    out = np.zeros_like(mean)
    positive = std > 1e-12
    z = (best - mean[positive]) / std[positive]
    out[positive] = (best - mean[positive]) * ndtr(z) + std[positive] * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    out[~positive] = np.maximum(best - mean[~positive], 0)
    return np.maximum(out, 0)


def gp_ei_search(log: OracleLog, candidates: np.ndarray) -> None:
    while len(log.values) < BUDGET:
        points = np.asarray(log.points, dtype=float)
        values = np.asarray(log.values, dtype=float)
        mean, std = gp_predict(points, values, candidates)
        standardized_best = (min(values) - float(np.mean(values))) / max(float(np.std(values)), 1e-8)
        acquisition = expected_improvement(standardized_best, mean, std)
        used = np.any(np.all(np.isclose(candidates[:, None, :], points[None, :, :], atol=1e-12), axis=2), axis=1)
        acquisition[used] = -np.inf
        index = int(np.argmax(acquisition))
        if used[index] or not np.isfinite(acquisition[index]):
            raise RuntimeError("代理模型没有新的可评价点")
        log.evaluate(candidates[index])


def evaluate_run(seed: int, method: str, candidates: np.ndarray) -> OracleLog:
    initial_rng = np.random.default_rng(seed)
    initial = initial_rng.uniform(size=(INITIAL, 2))
    method_rng = np.random.default_rng(seed + {"random": 1, "local": 2, "population": 3, "gp_ei": 4}[method] * 100_003)
    log = OracleLog(method, seed)
    use_initial(log, initial)
    if method == "random":
        random_search(log, method_rng)
    elif method == "local":
        local_search(log, method_rng)
    elif method == "population":
        population_search(log, method_rng)
    else:
        gp_ei_search(log, candidates)
    assert len(log.values) == BUDGET
    assert np.all(np.diff(log.best_so_far) <= 1e-12)
    assert np.allclose(log.values, branin_hidden_formula(np.asarray(log.points)), atol=1e-12)
    return log


def draw(runs: dict[str, list[OracleLog]], grid: np.ndarray, figure_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.45), layout="constrained")
    side = round(len(grid) ** 0.5)
    if side * side != len(grid):
        raise ValueError("候选网格不是正方形，不能直接画曲面")
    xx = -5 + 15 * grid[:, 0].reshape(side, side)
    yy = 15 * grid[:, 1].reshape(side, side)
    # 以下候选网格真值只供作者事后画图；没有一个传给搜索器。
    zz = branin_hidden_formula(grid).reshape(side, side)
    left.contourf(xx, yy, zz, levels=np.linspace(0, 200, 21), cmap="viridis", extend="max")
    left.contour(xx, yy, zz, levels=[0.5, 2, 10, 50], colors="white", linewidths=0.65)
    example = runs["gp_ei"][0]
    points = np.asarray(example.points)
    left.scatter(-5 + 15 * points[:, 0], 15 * points[:, 1], s=13, c=np.arange(BUDGET), cmap="plasma", edgecolors="black", linewidths=0.2)
    left.scatter(-5 + 15 * points[:INITIAL, 0], 15 * points[:INITIAL, 1], s=50, facecolors="none", edgecolors="white", linewidths=1)
    left.set(xlabel="旋钮 x1", ylabel="旋钮 x2", title="Branin 曲面与 GP-EI 一次轨迹（事后查看）")
    left.set_xlim(-5, 10)
    left.set_ylim(0, 15)

    colors = {"random": "#4e79a7", "local": "#f28e2b", "population": "#59a14f", "gp_ei": "#b07aa1"}
    display_names = {"random": "全域随机", "local": "局部重起", "population": "种群变异", "gp_ei": "GP 预期改进"}
    for method in METHODS:
        curves = np.asarray([run.best_so_far for run in runs[method]])
        median = np.median(curves, axis=0)
        q25, q75 = np.quantile(curves, [0.25, 0.75], axis=0)
        n = np.arange(1, BUDGET + 1)
        right.plot(n, median, color=colors[method], label=display_names[method])
        right.fill_between(n, q25, q75, color=colors[method], alpha=0.15)
    right.axhline(SUCCESS_VALUE, color="black", linestyle=":", linewidth=0.8)
    right.set(xlabel="真实评价次数（含 6 个共同起点）", ylabel="截至当前最佳已见值", title="12 次起点的中位数与中间一半", yscale="log")
    right.legend(fontsize=8)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定预算下比较黑箱优化；可单独加密 GP 候选网格")
    parser.add_argument("--grid-side", type=int, default=41, help="每条网格轴的点数")
    args = parser.parse_args()
    if args.grid_side < 3:
        parser.error("--grid-side 必须至少为 3")
    result_path = RESULT if args.grid_side == 41 else ROOT / "results" / f"black_box_budget_grid{args.grid_side}.json"
    figure_path = FIGURE if args.grid_side == 41 else ROOT / "figures" / f"black_box_budget_grid{args.grid_side}.png"
    # 两个可手算的 EI 边界：确定优于现值时是确定改进，均值恰在现值时只剩不确定项。
    assert abs(expected_improvement(1.0, np.array([0.6]), np.array([0.0]))[0] - 0.4) < 1e-12
    assert abs(expected_improvement(1.0, np.array([1.0]), np.array([2.0]))[0] - 2 / np.sqrt(2 * np.pi)) < 1e-12
    lin = np.linspace(0, 1, args.grid_side)
    candidates = np.array(np.meshgrid(lin, lin)).reshape(2, -1).T
    runs = {method: [evaluate_run(seed, method, candidates) for seed in SEEDS] for method in METHODS}
    summaries = {}
    for method in METHODS:
        final = np.asarray([run.best_so_far[-1] for run in runs[method]])
        summaries[method] = {
            "final_median": float(np.median(final)),
            "final_q25": float(np.quantile(final, 0.25)),
            "final_q75": float(np.quantile(final, 0.75)),
            "under_0_5": int(np.sum(final <= SUCCESS_VALUE)),
            "individual_final": final.tolist(),
        }
    analytic_minima_xy = np.asarray([[-np.pi, 12.275], [np.pi, 2.275], [3 * np.pi, 2.475]])
    analytic_minima_unit = np.column_stack(((analytic_minima_xy[:, 0] + 5) / 15, analytic_minima_xy[:, 1] / 15))
    posthoc_values = branin_hidden_formula(analytic_minima_unit)
    grid_values = branin_hidden_formula(candidates)
    grid_best_index = int(np.argmin(grid_values))
    assert np.allclose(posthoc_values, 10 / (8 * np.pi), atol=1e-12)
    record = {
        "purpose": "仅比较二维确定性 Branin 函数上本次实现和预算，不外推到所有黑箱或机器学习调参",
        "domain_in_code": "[0,1]^2; transformed to x1∈[-5,10], x2∈[0,15]",
        "algorithm_can_access": "previous points and values through OracleLog only",
        "budget_each_run": BUDGET,
        "common_initial_evaluations": INITIAL,
        "seeds": list(SEEDS),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "success_threshold": SUCCESS_VALUE,
        "gp_candidate_grid_side": args.grid_side,
        "summary": summaries,
        "author_posthoc_checks_not_seen_by_search": {
            "global_lower_bound": float(10 / (8 * np.pi)),
            "three_analytic_minimizers_xy": analytic_minima_xy.tolist(),
            "three_values": posthoc_values.tolist(),
            "gp_candidate_grid_best_unit": candidates[grid_best_index].tolist(),
            "gp_candidate_grid_best_value": float(grid_values[grid_best_index]),
        },
        "runs": {
            method: [
                {"seed": run.seed, "points_unit": run.points, "values": run.values, "best_so_far": run.best_so_far}
                for run in runs[method]
            ]
            for method in METHODS
        },
        "visualization_only_true_surface_calls": len(candidates),
    }
    result_path.parent.mkdir(exist_ok=True)
    figure_path.parent.mkdir(exist_ok=True)
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw(runs, candidates, figure_path)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"saved {result_path} and {figure_path}")


if __name__ == "__main__":
    main()
