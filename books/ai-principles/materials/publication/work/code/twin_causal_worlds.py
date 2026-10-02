"""C10：两个观察分布相同、干预效果不同的完整二值机制。"""

import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORK = Path(__file__).resolve().parents[1]
P_X_NOISE = 0.1
P_A_Y_NOISE = 0.1
P_B_Y_NOISE = 0.18
SEED = 20260924
SAMPLE_SIZE = 20000


def probability(bit, chance_one):
    return chance_one if bit else 1.0 - chance_one


def y_after(world, u, own_noise, forced_x):
    if world == "A":
        return u ^ own_noise
    if world == "B":
        return forced_x ^ own_noise
    raise ValueError(world)


def exact_world(world):
    chance_y_noise = P_A_Y_NOISE if world == "A" else P_B_Y_NOISE
    observed = defaultdict(float)
    observed_with_u = defaultdict(float)
    potential_with_u_x = defaultdict(float)
    under_do = {0: defaultdict(float), 1: defaultdict(float)}
    potentials = defaultdict(float)
    for u, nx, noise in itertools.product((0, 1), repeat=3):
        weight = (probability(u, 0.5)
                  * probability(nx, P_X_NOISE)
                  * probability(noise, chance_y_noise))
        x = u ^ nx
        y = y_after(world, u, noise, x)
        observed[(x, y)] += weight
        observed_with_u[(u, x, y)] += weight
        y0 = y_after(world, u, noise, 0)
        y1 = y_after(world, u, noise, 1)
        potential_with_u_x[(u, x, 0, y0)] += weight
        potential_with_u_x[(u, x, 1, y1)] += weight
        potentials[(y0, y1)] += weight
        under_do[0][y0] += weight
        under_do[1][y1] += weight

    joint = [[observed[(x, y)] for y in (0, 1)] for x in (0, 1)]
    p_y1_given_x = [
        observed[(x, 1)] / sum(observed[(x, y)] for y in (0, 1))
        for x in (0, 1)
    ]
    causal = [under_do[x][1] for x in (0, 1)]
    given_u = {
        str(u): {
            str(x): {
                "p_x_given_u": (
                    sum(observed_with_u[(u, x, y)] for y in (0, 1))
                    / sum(observed_with_u[(u, a, y)]
                          for a in (0, 1) for y in (0, 1))
                ),
                "p_y1_given_x_u": (
                    observed_with_u[(u, x, 1)]
                    / sum(observed_with_u[(u, x, y)] for y in (0, 1))
                ),
            }
            for x in (0, 1)
        }
        for u in (0, 1)
    }
    adjusted = [
        sum(
            given_u[str(u)][str(x)]["p_y1_given_x_u"] * 0.5
            for u in (0, 1)
        )
        for x in (0, 1)
    ]
    potential_y1_by_u_and_observed_x = {
        str(u): {
            str(x): {
                str(forced): (
                    potential_with_u_x[(u, x, forced, 1)]
                    / sum(observed_with_u[(u, x, y)] for y in (0, 1))
                )
                for forced in (0, 1)
            }
            for x in (0, 1)
        }
        for u in (0, 1)
    }
    return {
        "observed_joint_rows_x_columns_y": joint,
        "p_y1_given_x": p_y1_given_x,
        "observed_gap": p_y1_given_x[1] - p_y1_given_x[0],
        "p_y1_under_do_x": causal,
        "intervention_average_effect": causal[1] - causal[0],
        "given_u": given_u,
        "adjusted_p_y1_under_do_x": adjusted,
        "p_potential_y1_given_u_and_observed_x": potential_y1_by_u_and_observed_x,
        "potential_outcome_joint_rows_y0_columns_y1": [
            [potentials[(y0, y1)] for y1 in (0, 1)]
            for y0 in (0, 1)
        ],
        "potential_outcome_average_difference": sum(
            (y1 - y0) * weight
            for (y0, y1), weight in potentials.items()
        ),
    }


def finite_sample(world):
    chance_y_noise = P_A_Y_NOISE if world == "A" else P_B_Y_NOISE
    rng = random.Random(SEED)
    seen_counts = [0, 0]
    seen_success = [0, 0]
    assigned_counts = [0, 0]
    assigned_success = [0, 0]
    for _ in range(SAMPLE_SIZE):
        u = int(rng.random() < 0.5)
        nx = int(rng.random() < P_X_NOISE)
        noise = int(rng.random() < chance_y_noise)
        x_seen = u ^ nx
        y_seen = y_after(world, u, noise, x_seen)
        seen_counts[x_seen] += 1
        seen_success[x_seen] += y_seen

        assignment = int(rng.random() < 0.5)  # 与 U 和噪声独立
        y_assigned = y_after(world, u, noise, assignment)
        assigned_counts[assignment] += 1
        assigned_success[assignment] += y_assigned
    observed_rates = [
        seen_success[x] / seen_counts[x] for x in (0, 1)
    ]
    assigned_rates = [
        assigned_success[x] / assigned_counts[x] for x in (0, 1)
    ]
    return {
        "n": SAMPLE_SIZE,
        "seed": SEED,
        "observation_counts_by_x": seen_counts,
        "observation_p_y1_given_x": observed_rates,
        "observation_gap": observed_rates[1] - observed_rates[0],
        "random_assignment_counts_by_x": assigned_counts,
        "random_assignment_p_y1": assigned_rates,
        "random_assignment_gap": assigned_rates[1] - assigned_rates[0],
        "scope": "One simulation from a stipulated mechanism; not data from real people or a real randomized trial.",
    }


def near(actual, expected):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"精确核算得到 {actual}，预定应为 {expected}")


def verify_exact(a, b):
    expected_joint = [[0.41, 0.09], [0.09, 0.41]]
    for result in (a, b):
        for x in (0, 1):
            for y in (0, 1):
                near(result["observed_joint_rows_x_columns_y"][x][y],
                     expected_joint[x][y])
        for actual, expected in zip(result["p_y1_given_x"], [0.18, 0.82]):
            near(actual, expected)
        near(result["observed_gap"], 0.64)
        for x in (0, 1):
            near(result["adjusted_p_y1_under_do_x"][x],
                 result["p_y1_under_do_x"][x])
        near(result["intervention_average_effect"],
             result["potential_outcome_average_difference"])
        for u in (0, 1):
            for x in (0, 1):
                if result["given_u"][str(u)][str(x)]["p_x_given_u"] <= 0:
                    raise ValueError("给定 U 后某一种 X 没有正机会")
            for forced in (0, 1):
                left = result["p_potential_y1_given_u_and_observed_x"][str(u)]["0"][str(forced)]
                right = result["p_potential_y1_given_u_and_observed_x"][str(u)]["1"][str(forced)]
                near(left, right)
    for actual in a["p_y1_under_do_x"]:
        near(actual, 0.5)
    near(a["intervention_average_effect"], 0.0)
    for actual, expected in zip(b["p_y1_under_do_x"], [0.18, 0.82]):
        near(actual, expected)
    near(b["intervention_average_effect"], 0.64)


def draw_worlds(a, b):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, name, result in zip(axes, ("世界 A：共同条件", "世界 B：X 进入 Y 的生成式"), (a, b)):
        ax.bar([0, 1], result["p_y1_given_x"], color="#24527a",
               label="观察条件")
        ax.bar([3, 4], result["p_y1_under_do_x"], color="#a54d13",
               label="外部设定")
        ax.set_xticks([0, 1, 3, 4], ["看见 X=0", "看见 X=1", "设定 X=0", "设定 X=1"])
        ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("P(Y=1)")
    fig.suptitle("同一张 X、Y 观察表，仍可能有不同的干预答案",
                 fontsize=12)
    fig.tight_layout()
    path = WORK / "figures/twin_causal_worlds.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main():
    a, b = exact_world("A"), exact_world("B")
    verify_exact(a, b)
    sample_a, sample_b = finite_sample("A"), finite_sample("B")
    result = {
        "protocol": "work/verification/C10_twin_worlds_protocol.md",
        "kind": "author-constructed causal mechanism, exact enumeration plus one seeded simulation",
        "exogenous_probabilities": {
            "p_u_one": 0.5, "p_nx_one": P_X_NOISE,
            "p_world_a_ny_one": P_A_Y_NOISE,
            "p_world_b_nb_one": P_B_Y_NOISE,
        },
        "world_a": {"exact": a, "one_simulation": sample_a},
        "world_b": {"exact": b, "one_simulation": sample_b},
        "figure": "work/figures/twin_causal_worlds.png",
        "scope": "Both worlds yield exactly the same P(X,Y), but do(X) has different effects. U is not present in the X-Y-only table. Simulation is instructional, not an empirical causal claim.",
    }
    folder = WORK / "results"
    folder.mkdir(exist_ok=True)
    (folder / "twin_causal_worlds.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    draw_worlds(a, b)
    print("两世界共同观察表:", a["observed_joint_rows_x_columns_y"])
    print("共同观察差:", a["observed_gap"])
    print("A/B 真正干预差:", a["intervention_average_effect"],
          b["intervention_average_effect"])
    print("一次模拟的观察差:", sample_a["observation_gap"],
          sample_b["observation_gap"])
    print("一次模拟的随机分配差:", sample_a["random_assignment_gap"],
          sample_b["random_assignment_gap"])


if __name__ == "__main__":
    main()
