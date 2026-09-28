"""C24: one held-out CartPole evaluation and same-image/different-motion probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from cartpole_dqn import QNetwork


ROOT = Path(__file__).resolve().parents[2]


def run_one_policy(kind: str, seeds: list[int], model: QNetwork | None, rng_seed: int) -> dict:
    env = gym.make("CartPole-v1")
    rng = random.Random(rng_seed)
    lengths = []
    ended = []
    try:
        for seed in seeds:
            state, _ = env.reset(seed=seed)
            length = 0
            while True:
                if kind == "dqn":
                    assert model is not None
                    with torch.no_grad():
                        q = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                        action = int(q.argmax(dim=1).item())
                elif kind == "random":
                    action = rng.randrange(2)
                elif kind == "angle_velocity_rule":
                    # A plausible hand rule, fixed before reading held-out results.
                    action = int(float(state[2]) + 0.1 * float(state[3]) > 0)
                else:
                    raise ValueError(kind)
                state, _, terminated, truncated, _ = env.step(action)
                length += 1
                if terminated or truncated:
                    lengths.append(length)
                    ended.append("physical" if terminated else "time_limit")
                    break
    finally:
        env.close()
    return {
        "lengths": lengths,
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "reached_500_step_limit": sum(length == 500 for length in lengths),
        "ended": ended,
    }


def same_visible_different_motion() -> dict:
    env = gym.make("CartPole-v1")
    cases = []
    try:
        for velocity, angular_velocity in ((0.5, 0.5), (-0.5, -0.5)):
            env.reset(seed=123)
            # Controlled simulator probe only: unwrapped.state is an internal
            # Gymnasium 1.3.0 variable, not the public agent observation API.
            env.unwrapped.state = (0.0, velocity, 0.03, angular_velocity)
            next_observation, reward, terminated, truncated, _ = env.step(1)
            cases.append({
                "state_before": [0.0, velocity, 0.03, angular_velocity],
                "visible_if_velocities_hidden": [0.0, 0.03],
                "action": 1,
                "next_observation": np.asarray(next_observation, dtype=float).tolist(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    finally:
        env.close()
    if cases[0]["visible_if_velocities_hidden"] != cases[1]["visible_if_velocities_hidden"]:
        raise AssertionError("same reduced observation expected")
    if cases[0]["next_observation"] == cases[1]["next_observation"]:
        raise AssertionError("opposite velocities should change the next state")
    return {
        "scope": "Gymnasium 1.3.0 internal state mutation for one-step diagnostic only; training and held-out test use public reset/step observations",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-report", type=Path, default=Path("work/runs/c24_dqn_trial400.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("work/results/c24_cartpole_test.json")
    )
    args = parser.parse_args()
    report_path = args.training_report if args.training_report.is_absolute() else ROOT / args.training_report
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"preserve held-out result; choose a new --out: {out}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_info = report["best_checkpoint"]
    model_path = ROOT / checkpoint_info["path"]
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_sha != checkpoint_info["sha256"]:
        raise RuntimeError("checkpoint differs from training report")
    if report["gymnasium_version"] != gym.__version__:
        raise RuntimeError("Gymnasium version changed after training")

    torch.set_num_threads(1)
    model = QNetwork()
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    seeds = list(range(20_000, 20_050))
    if set(seeds) & set(report["validation_seeds"]):
        raise AssertionError("test and validation seeds overlap")
    policies = {
        "dqn": run_one_policy("dqn", seeds, model, 0),
        "random": run_one_policy("random", seeds, None, 66),
        "angle_velocity_rule": run_one_policy("angle_velocity_rule", seeds, None, 0),
    }
    output = {
        "scope": "single opening of 50 disjoint CartPole-v1 reset seeds after validation-selected model is fixed; no model or hyperparameter choice made from this file",
        "gymnasium_version": gym.__version__,
        "training_report": str(report_path.relative_to(ROOT)),
        "training_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "best_model": checkpoint_info,
        "test_seeds": seeds,
        "policies": policies,
        "same_reduced_observation_different_motion": same_visible_different_motion(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out)
    for name, result in policies.items():
        print(name, f"mean={result['mean']:.2f}", "median=", result["median"], "at500=", result["reached_500_step_limit"])
    print("same reduced observation next-angle pair:", [
        row["next_observation"][2]
        for row in output["same_reduced_observation_different_motion"]["cases"]
    ])


if __name__ == "__main__":
    main()
