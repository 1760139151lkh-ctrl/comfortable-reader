"""C21 finite decision laboratory: exact trajectory enumeration and sampled runs.

The fictional courier environment is a teaching model, not a measurement of a
physical battery or a trained language model. The controller sees exact energy
for the two policy comparisons. A separate lamp projection demonstrates what
would break if only a coarse observation were available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ("slow", "express", "charge")
POLICIES = ("immediate", "conserve")


def outcomes(energy: int, action: str) -> tuple[tuple[float, int, int], ...]:
    """Return (probability, reward, next energy), after one action."""
    if energy not in (0, 1, 2) or action not in ACTIONS:
        raise ValueError("energy must be 0/1/2 and action slow/express/charge")
    if action == "charge":
        return ((1.0, -1, 2),)
    if action == "slow":
        return ((1.0, 2, energy - 1),) if energy >= 1 else ((1.0, -3, 0),)
    if energy == 2:
        return ((0.75, 4, 0), (0.25, -2, 0))
    return ((1.0, -4, energy),)


def lamp(energy: int) -> str:
    return "red" if energy == 0 else "green"


def choose(policy: str, energy: int) -> str:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if energy == 0:
        return "charge"
    if energy == 1:
        return "slow"
    return "express" if policy == "immediate" else "slow"


def enumerate_paths(policy: str, horizon: int, start: int = 2) -> list[dict]:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    paths = [{"probability": 1.0, "energy": start, "return": 0, "events": []}]
    for t in range(horizon):
        next_paths = []
        for path in paths:
            energy = path["energy"]
            action = choose(policy, energy)
            for prob, reward, next_energy in outcomes(energy, action):
                event = {
                    "t": t,
                    "state_before": energy,
                    "observation_before": lamp(energy),
                    "action": action,
                    "reward_index": t + 1,
                    "reward": reward,
                    "state_after": next_energy,
                }
                next_paths.append({
                    "probability": path["probability"] * prob,
                    "energy": next_energy,
                    "return": path["return"] + reward,
                    "events": path["events"] + [event],
                })
        paths = next_paths
    return paths


def exact_result(policy: str, horizon: int) -> dict:
    paths = enumerate_paths(policy, horizon)
    mass = sum(path["probability"] for path in paths)
    if not math.isclose(mass, 1.0, abs_tol=1e-12):
        raise AssertionError(f"trajectory probability mass is {mass}")
    mean = sum(path["probability"] * path["return"] for path in paths)
    variance = sum(
        path["probability"] * (path["return"] - mean) ** 2 for path in paths
    )
    return {
        "policy": policy,
        "horizon": horizon,
        "expected_return": mean,
        "return_variance": variance,
        "paths": paths,
    }


def run_once(policy: str, horizon: int, rng: random.Random) -> tuple[int, list[dict]]:
    energy = 2
    total = 0
    history = []
    for t in range(horizon):
        action = choose(policy, energy)
        branches = outcomes(energy, action)
        if len(branches) == 1:
            _, reward, next_energy = branches[0]
        else:
            draw = rng.random()
            cutoff = 0.0
            for prob, reward, next_energy in branches:
                cutoff += prob
                if draw < cutoff:
                    break
        history.append({
            "t": t,
            "state_before": energy,
            "observation_before": lamp(energy),
            "action": action,
            "reward_index": t + 1,
            "reward": reward,
            "state_after": next_energy,
        })
        total += reward
        energy = next_energy
    return total, history


def simulate(policy: str, horizon: int, episodes: int, seed: int) -> dict:
    rng = random.Random(seed)
    total = 0
    square_total = 0
    example = None
    for _ in range(episodes):
        value, history = run_once(policy, horizon, rng)
        if example is None:
            example = history
        total += value
        square_total += value * value
    mean = total / episodes
    variance = max(0.0, square_total / episodes - mean * mean)
    return {
        "episodes": episodes,
        "seed": seed,
        "sample_mean_return": mean,
        "sample_variance": variance,
        "sample_standard_error": math.sqrt(variance / episodes),
        "first_trajectory": example,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--out", type=Path, default=Path("work/results/c21_delivery_lab.json"))
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"keep previous result; choose a different --out: {out}")
    checks = []
    for horizon in (1, 2, 3):
        for policy in POLICIES:
            exact = exact_result(policy, horizon)
            sampled = simulate(
                policy, horizon, args.episodes,
                args.seed + 100 * horizon + (0 if policy == "immediate" else 1),
            )
            checks.append({**exact, "sampled": sampled})
    expected = {
        (row["policy"], row["horizon"]): row["expected_return"] for row in checks
    }
    assert expected["immediate", 1] == 2.5
    assert expected["conserve", 1] == 2.0
    assert expected["immediate", 2] == 1.5
    assert expected["conserve", 2] == 4.0
    assert expected["immediate", 3] == 4.0
    assert expected["conserve", 3] == 3.0
    output = {
        "scope": "fictional three-level courier energy model; exact enumeration versus independently sampled episodes; no real robot, GPU, or language-model training",
        "start_energy": 2,
        "states": [0, 1, 2],
        "actions": list(ACTIONS),
        "reward_timing": "state_t -> action_t -> (reward_{t+1}, state_{t+1})",
        "transition_table": {
            str(energy): {
                action: [
                    {"probability": prob, "reward": reward, "next_energy": next_energy}
                    for prob, reward, next_energy in outcomes(energy, action)
                ]
                for action in ACTIONS
            }
            for energy in (0, 1, 2)
        },
        "lamp_alias": {
            "after_charge": {"energy": 2, "observation": lamp(2),
                             "express_outcomes": outcomes(2, "express")},
            "after_slow_from_2": {"energy": 1, "observation": lamp(1),
                                  "express_outcomes": outcomes(1, "express")},
        },
        "policies": {
            "immediate": {"energy_0": "charge", "energy_1": "slow",
                          "energy_2": "express"},
            "conserve": {"energy_0": "charge", "energy_1": "slow",
                         "energy_2": "slow"},
        },
        "comparisons": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out)
    for horizon in (1, 2, 3):
        rows = [row for row in checks if row["horizon"] == horizon]
        print(
            f"steps={horizon} " +
            " ".join(
                f"{row['policy']}: exact={row['expected_return']:.3f}, "
                f"sample={row['sampled']['sample_mean_return']:.3f}"
                for row in rows
            )
        )


if __name__ == "__main__":
    main()
