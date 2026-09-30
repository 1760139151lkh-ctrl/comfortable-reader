"""C24: learn a one-step CartPole model, inspect rollout error, and plan briefly.

One linear dynamics model is deliberately simple. Exact simulator equations
are never passed to fitting or planning. Test seeds are opened only after
the model, candidate horizon, and hand-designed planning cost are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from cartpole_dqn import QNetwork
from cartpole_evaluate import run_one_policy


ROOT = Path(__file__).resolve().parents[2]


def hand_action(state: np.ndarray) -> int:
    return int(float(state[2]) + 0.1 * float(state[3]) > 0)


def collect_random(episode_seeds: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = gym.make("CartPole-v1")
    states, actions, next_states = [], [], []
    try:
        for seed in episode_seeds:
            rng = random.Random(seed + 91)
            state, _ = env.reset(seed=seed)
            while True:
                action = rng.randrange(2)
                next_state, _, terminated, truncated, _ = env.step(action)
                states.append(np.asarray(state, dtype=np.float64))
                actions.append(action)
                next_states.append(np.asarray(next_state, dtype=np.float64))
                state = next_state
                if terminated or truncated:
                    break
    finally:
        env.close()
    return np.stack(states), np.asarray(actions, dtype=np.int64), np.stack(next_states)


def full_features(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    force = (2 * actions - 1).astype(np.float64)[:, None]
    return np.concatenate([states, force, np.ones((len(states), 1))], axis=1)


def reduced_features(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    force = (2 * actions - 1).astype(np.float64)[:, None]
    return np.concatenate([states[:, (0, 2)], force, np.ones((len(states), 1))], axis=1)


def fit_models(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Full model predicts a delta; reduced model has no velocity inputs and
    # predicts the next full state for an information-loss diagnostic.
    full, *_ = np.linalg.lstsq(full_features(states, actions), next_states - states, rcond=None)
    reduced, *_ = np.linalg.lstsq(reduced_features(states, actions), next_states, rcond=None)
    return full, reduced


def predict_full(state: np.ndarray, action: int, coefficients: np.ndarray) -> np.ndarray:
    features = np.asarray([*state, 2 * action - 1, 1.0], dtype=np.float64)
    return state + features @ coefficients


def one_step_errors(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    full: np.ndarray, reduced: np.ndarray,
) -> dict:
    full_prediction = states + full_features(states, actions) @ full
    reduced_prediction = reduced_features(states, actions) @ reduced
    return {
        "transitions": len(states),
        "full_rmse_by_component": np.sqrt(np.mean((full_prediction - next_states) ** 2, axis=0)).tolist(),
        "reduced_rmse_by_component": np.sqrt(np.mean((reduced_prediction - next_states) ** 2, axis=0)).tolist(),
        "component_order": ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"],
    }


def collect_heldout_hand_sequences(seeds: list[int], steps: int) -> list[dict]:
    env = gym.make("CartPole-v1")
    sequences = []
    try:
        for seed in seeds:
            state, _ = env.reset(seed=seed)
            states = [np.asarray(state, dtype=np.float64)]
            actions = []
            for _ in range(steps):
                action = hand_action(np.asarray(state))
                state, _, terminated, truncated, _ = env.step(action)
                actions.append(action)
                states.append(np.asarray(state, dtype=np.float64))
                if terminated or truncated:
                    break
            sequences.append({"seed": seed, "states": states, "actions": actions})
    finally:
        env.close()
    return sequences


def rollout_error(sequences: list[dict], model: np.ndarray) -> dict:
    horizons = (1, 2, 5, 10, 20, 50)
    angle_errors = {h: [] for h in horizons}
    position_errors = {h: [] for h in horizons}
    teacher_forced_angle_errors = []
    for sequence in sequences:
        states, actions = sequence["states"], sequence["actions"]
        predicted = np.asarray(states[0], dtype=np.float64)
        for step, action in enumerate(actions, start=1):
            one_step = predict_full(np.asarray(states[step - 1]), action, model)
            teacher_forced_angle_errors.append(abs(float(one_step[2] - states[step][2])))
            predicted = predict_full(predicted, action, model)
            if step in angle_errors:
                angle_errors[step].append(abs(float(predicted[2] - states[step][2])))
                position_errors[step].append(abs(float(predicted[0] - states[step][0])))
    return {
        "meaning": "same true action sequence fed open-loop into learned model; after step 1, predicted states feed the next model step; teacher-forced comparison always starts from true state",
        "horizons": {
            h: {
                "sequences_reaching_h": len(angle_errors[h]),
                "mean_abs_pole_angle_error_radians": statistics.fmean(angle_errors[h]) if angle_errors[h] else None,
                "mean_abs_cart_position_error": statistics.fmean(position_errors[h]) if position_errors[h] else None,
            }
            for h in horizons
        },
        "teacher_forced_mean_abs_angle_error_radians": statistics.fmean(teacher_forced_angle_errors),
    }


def predicted_sequence_cost(start: np.ndarray, actions: tuple[int, ...], model: np.ndarray) -> float:
    state = np.asarray(start, dtype=np.float64)
    cost = 0.0
    for index, action in enumerate(actions):
        state = predict_full(state, action, model)
        x, x_velocity, angle, angle_velocity = state
        if abs(x) > 2.4 or abs(angle) > 0.2095:
            return cost + 100.0 + (len(actions) - index)
        cost += 10.0 * angle * angle + 0.1 * angle_velocity * angle_velocity
        cost += 0.01 * x * x + 0.001 * x_velocity * x_velocity
    return cost


def plan_action(state: np.ndarray, model: np.ndarray, candidates: list[tuple[int, ...]]) -> int:
    best = min(candidates, key=lambda actions: predicted_sequence_cost(state, actions, model))
    return best[0]


def evaluate_receding_horizon(seeds: list[int], model: np.ndarray, horizon: int) -> dict:
    candidates = list(itertools.product((0, 1), repeat=horizon))
    env = gym.make("CartPole-v1")
    lengths = []
    ended = []
    try:
        for seed in seeds:
            state, _ = env.reset(seed=seed)
            length = 0
            while True:
                action = plan_action(np.asarray(state, dtype=np.float64), model, candidates)
                state, _, terminated, truncated, _ = env.step(action)
                length += 1
                if terminated or truncated:
                    lengths.append(length)
                    ended.append("physical" if terminated else "time_limit")
                    break
    finally:
        env.close()
    return {
        "horizon": horizon,
        "candidates_at_each_real_step": len(candidates),
        "planning_objective": "hand-designed sum of angle/angle-speed/cart-position/cart-speed squared, plus predicted threshold penalty; not a learned reward or the exact Gymnasium +1 objective",
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "reached_500_step_limit": sum(x == 500 for x in lengths),
        "lengths": lengths,
        "ended": ended,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("work/results/c24_cartpole_world_model.json")
    )
    parser.add_argument(
        "--dqn-model", type=Path, default=Path("work/runs/c24_dqn_trial400.pt")
    )
    parser.add_argument(
        "--dqn-report", type=Path, default=Path("work/runs/c24_dqn_trial400.json")
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"keep old result; choose another --out: {out}")
    model_path = args.dqn_model if args.dqn_model.is_absolute() else ROOT / args.dqn_model
    dqn_report_path = args.dqn_report if args.dqn_report.is_absolute() else ROOT / args.dqn_report
    dqn_report = json.loads(dqn_report_path.read_text(encoding="utf-8"))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_sha != dqn_report["best_checkpoint"]["sha256"]:
        raise RuntimeError("DQN checkpoint differs from its training report")

    train_seeds = list(range(30_000, 30_400))
    validation_seeds = list(range(40_000, 40_050))
    test_seeds = list(range(60_000, 60_030))
    if set(train_seeds) & set(validation_seeds) or set(train_seeds) & set(test_seeds):
        raise AssertionError("data split overlap")
    train_states, train_actions, train_next = collect_random(train_seeds)
    full, reduced = fit_models(train_states, train_actions, train_next)
    validation = collect_random(validation_seeds)
    validation_error = one_step_errors(*validation, full, reduced)
    hand_sequences = collect_heldout_hand_sequences(test_seeds, steps=50)
    open_loop = rollout_error(hand_sequences, full)
    receding = evaluate_receding_horizon(test_seeds, full, horizon=5)

    torch.set_num_threads(1)
    dqn = QNetwork()
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    dqn.load_state_dict(saved["state_dict"])
    dqn.eval()
    test_comparisons = {
        "dqn": run_one_policy("dqn", test_seeds, dqn, 0),
        "hand_angle_velocity_rule": run_one_policy("angle_velocity_rule", test_seeds, None, 0),
        "random": run_one_policy("random", test_seeds, None, 77),
    }
    result = {
        "scope": "Gymnasium 1.3.0 CartPole-v1 numerical-state model; one linear next-state approximation from random actions, one open-loop test, and a hand-cost receding-horizon planner; not MuZero/PlaNet",
        "gymnasium_version": gym.__version__,
        "train_episode_seeds": [train_seeds[0], train_seeds[-1]],
        "train_episode_count": len(train_seeds),
        "train_transition_count": len(train_states),
        "one_step_validation_seeds": validation_seeds,
        "one_step_validation": validation_error,
        "unseen_control_and_rollout_seeds": test_seeds,
        "open_loop_heldout_hand_rule": open_loop,
        "receding_horizon_planner": receding,
        "same_seed_comparators": test_comparisons,
        "linear_full_delta_coefficients": full.tolist(),
        "linear_reduced_next_coefficients": reduced.tolist(),
        "feature_order_full": ["x", "x_velocity", "pole_angle", "angle_velocity", "action_force_minus_or_plus_one", "bias"],
        "feature_order_reduced": ["x", "pole_angle", "action_force_minus_or_plus_one", "bias"],
        "dqn_model_path": str(model_path.relative_to(ROOT)),
        "dqn_model_sha256": model_sha,
        "dqn_training_report": str(dqn_report_path.relative_to(ROOT)),
        "dqn_training_report_sha256": hashlib.sha256(dqn_report_path.read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out)
    print("one-step angle RMSE, full/reduced:", validation_error["full_rmse_by_component"][2], validation_error["reduced_rmse_by_component"][2])
    print("open-loop mean abs angle error:", {
        h: round(v["mean_abs_pole_angle_error_radians"], 6) if v["mean_abs_pole_angle_error_radians"] is not None else None
        for h, v in open_loop["horizons"].items()
    })
    print("mean episodes, learned-model MPC / DQN / hand rule / random:",
          receding["mean"], test_comparisons["dqn"]["mean"],
          test_comparisons["hand_angle_velocity_rule"]["mean"], test_comparisons["random"]["mean"])


if __name__ == "__main__":
    main()
