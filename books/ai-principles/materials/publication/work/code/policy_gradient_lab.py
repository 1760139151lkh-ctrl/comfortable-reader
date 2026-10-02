"""C23: probability gradients and a small actor-critic on the C21 courier.

The sampler sees C21 outcomes, while parameter updates see only sampled
events. Exact model calculations below are diagnostics, never update targets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path

from delivery_decision_lab import ACTIONS, choose, outcomes
from value_learning_lab import STATES, action_return, finite_model


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = (0, 1, 10, 100, 500, 1000, 2000)


def sigmoid(theta: float) -> float:
    if theta >= 0:
        z = math.exp(-theta)
        return 1 / (1 + z)
    z = math.exp(theta)
    return z / (1 + z)


def two_step_branches() -> list[dict]:
    """First action stochastic; thereafter C21's fixed immediate policy."""
    branches = []
    for action in ("express", "slow"):
        for p1, r1, s1 in outcomes(2, action):
            second = choose("immediate", s1)
            for p2, r2, _ in outcomes(s1, second):
                branches.append(
                    {
                        "first_action": action,
                        "first_reward": r1,
                        "second_action": second,
                        "second_reward": r2,
                        "conditional_probability": p1 * p2,
                        "return": r1 + r2,
                    }
                )
    for action in ("express", "slow"):
        total = sum(
            row["conditional_probability"]
            for row in branches
            if row["first_action"] == action
        )
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise AssertionError("incomplete two-step consequences")
    return branches


def score(action: str, p_express: float) -> float:
    # d log p(action) / d theta when p_express = sigmoid(theta).
    return (1.0 - p_express) if action == "express" else -p_express


def scalar_exact(theta: float, branches: list[dict]) -> dict:
    p = sigmoid(theta)
    objective = 0.0
    gradient = 0.0
    rows = []
    for row in branches:
        action = row["first_action"]
        action_probability = p if action == "express" else 1 - p
        joint_probability = action_probability * row["conditional_probability"]
        contribution = joint_probability * row["return"] * score(action, p)
        objective += joint_probability * row["return"]
        gradient += contribution
        rows.append(
            {
                **row,
                "policy_probability": action_probability,
                "joint_probability": joint_probability,
                "score": score(action, p),
                "gradient_contribution": contribution,
            }
        )

    def moments(baseline: float) -> dict:
        mean = sum(
            row["joint_probability"] * (row["return"] - baseline) * row["score"]
            for row in rows
        )
        second = sum(
            row["joint_probability"]
            * ((row["return"] - baseline) * row["score"]) ** 2
            for row in rows
        )
        return {"mean": mean, "variance": second - mean * mean}

    return {
        "theta": theta,
        "p_express": p,
        "objective": objective,
        "direct_gradient": -2.5 * p * (1 - p),
        "score_gradient": gradient,
        "branches": rows,
        "zero_baseline_estimator": moments(0.0),
        "objective_baseline_estimator": moments(objective),
    }


def draw_outcome(energy: int, action: str, rng: random.Random) -> tuple[int, int]:
    draw = rng.random()
    cutoff = 0.0
    for probability, reward, next_energy in outcomes(energy, action):
        cutoff += probability
        if draw < cutoff:
            return reward, next_energy
    raise AssertionError("outcome probabilities do not sum to one")


def sample_two_step(
    target_p: float, behavior_p: float, baseline: float, samples: int, seed: int
) -> dict:
    rng = random.Random(seed)
    plain, centered, corrected_plain, corrected_centered = [], [], [], []
    action_counts = {"express": 0, "slow": 0}
    for _ in range(samples):
        first = "express" if rng.random() < behavior_p else "slow"
        r1, s1 = draw_outcome(2, first, rng)
        second = choose("immediate", s1)
        r2, _ = draw_outcome(s1, second, rng)
        actual_return = r1 + r2
        action_counts[first] += 1
        log_derivative = score(first, target_p)
        target_probability = target_p if first == "express" else 1 - target_p
        behavior_probability = behavior_p if first == "express" else 1 - behavior_p
        ratio = target_probability / behavior_probability
        plain.append(actual_return * log_derivative)
        centered.append((actual_return - baseline) * log_derivative)
        corrected_plain.append(ratio * actual_return * log_derivative)
        corrected_centered.append(ratio * (actual_return - baseline) * log_derivative)
    return {
        "target_p_express": target_p,
        "behavior_p_express": behavior_p,
        "samples": samples,
        "seed": seed,
        "action_counts": action_counts,
        "plain_mean": statistics.fmean(plain),
        "centered_mean": statistics.fmean(centered),
        "importance_corrected_plain_mean": statistics.fmean(corrected_plain),
        "importance_corrected_centered_mean": statistics.fmean(corrected_centered),
        "plain_variance": statistics.pvariance(plain),
        "centered_variance": statistics.pvariance(centered),
    }


def off_policy_exact(target_p: float, behavior_p: float, branches: list[dict]) -> dict:
    raw = 0.0
    corrected = 0.0
    centered_raw = 0.0
    centered_corrected = 0.0
    target_baseline = 4 - 2.5 * target_p
    for row in branches:
        action = row["first_action"]
        b = behavior_p if action == "express" else 1 - behavior_p
        t = target_p if action == "express" else 1 - target_p
        base = b * row["conditional_probability"] * row["return"] * score(action, target_p)
        raw += base
        corrected += base * t / b
        centered_base = (
            b * row["conditional_probability"]
            * (row["return"] - target_baseline) * score(action, target_p)
        )
        centered_raw += centered_base
        centered_corrected += centered_base * t / b
    return {
        "target_p_express": target_p,
        "behavior_p_express": behavior_p,
        "target_objective_baseline": target_baseline,
        "uncorrected_expected_score": raw,
        "importance_corrected_expected_score": corrected,
        "uncorrected_centered_expected_score": centered_raw,
        "importance_corrected_centered_expected_score": centered_corrected,
    }


def clipped_probe(old_p: float, new_p: float, epsilon: float, branches: list[dict]) -> dict:
    old_j = 4 - 2.5 * old_p
    new_j = 4 - 2.5 * new_p
    terms = []
    for row in branches:
        action = row["first_action"]
        old_action_p = old_p if action == "express" else 1 - old_p
        new_action_p = new_p if action == "express" else 1 - new_p
        ratio = new_action_p / old_action_p
        advantage = row["return"] - old_j
        unclipped = ratio * advantage
        clipped_ratio = min(max(ratio, 1 - epsilon), 1 + epsilon)
        pessimistic = min(unclipped, clipped_ratio * advantage)
        terms.append(
            {
                "first_action": action,
                "return": row["return"],
                "old_probability": old_action_p * row["conditional_probability"],
                "advantage_from_old_mean": advantage,
                "ratio": ratio,
                "unclipped_term": unclipped,
                "clipped_surrogate_term": pessimistic,
            }
        )
    old_kl_new = old_p * math.log(old_p / new_p) + (1 - old_p) * math.log(
        (1 - old_p) / (1 - new_p)
    )
    return {
        "old_p_express": old_p,
        "new_p_express": new_p,
        "epsilon": epsilon,
        "old_objective": old_j,
        "new_objective": new_j,
        "true_change": new_j - old_j,
        "old_to_new_kl": old_kl_new,
        "unclipped_surrogate": sum(x["old_probability"] * x["unclipped_term"] for x in terms),
        "clipped_surrogate": sum(
            x["old_probability"] * x["clipped_surrogate_term"] for x in terms
        ),
        "terms": terms,
    }


def softmax(logits: dict[str, float]) -> dict[str, float]:
    maximum = max(logits.values())
    weights = {a: math.exp(logits[a] - maximum) for a in ACTIONS}
    total = sum(weights.values())
    return {a: weights[a] / total for a in ACTIONS}


def blank_actor() -> dict[int, dict[int, dict[str, float]]]:
    return {
        n: {s: {a: 0.0 for a in ACTIONS} for s in STATES}
        for n in range(1, 4)
    }


def blank_critic() -> dict[int, dict[int, float]]:
    return {n: {s: 0.0 for s in STATES} for n in range(4)}


def sampled_actor_episode(
    actor: dict[int, dict[int, dict[str, float]]],
    start_energy: int,
    horizon: int,
    rng: random.Random,
) -> list[dict]:
    energy = start_energy
    events = []
    for n in range(horizon, 0, -1):
        probabilities = softmax(actor[n][energy])
        draw = rng.random()
        cutoff = 0.0
        chosen = ACTIONS[-1]
        for action in ACTIONS:
            cutoff += probabilities[action]
            if draw < cutoff:
                chosen = action
                break
        reward, next_energy = draw_outcome(energy, chosen, rng)
        events.append(
            {
                "remaining_actions": n,
                "state_before": energy,
                "action": chosen,
                "reward": reward,
                "state_after": next_energy,
                "behavior_probabilities": probabilities,
            }
        )
        energy = next_energy
    return events


def actor_critic_update(
    events: list[dict],
    actor: dict[int, dict[int, dict[str, float]]],
    critic: dict[int, dict[int, float]],
    actor_rate: float,
    critic_rate: float,
) -> None:
    # Events were sampled before this episode's actor update. Hold each sampled
    # behavior distribution fixed while applying its score-function gradient.
    for event in events:
        n = event["remaining_actions"]
        state = event["state_before"]
        next_state = event["state_after"]
        action = event["action"]
        delta = event["reward"] + critic[n - 1][next_state] - critic[n][state]
        critic[n][state] += critic_rate * delta
        probabilities = event["behavior_probabilities"]
        for candidate in ACTIONS:
            log_gradient = float(candidate == action) - probabilities[candidate]
            actor[n][state][candidate] += actor_rate * delta * log_gradient


def exact_policy_values(actor: dict[int, dict[int, dict[str, float]]]) -> dict:
    values = [{s: 0.0 for s in STATES}]
    for n in range(1, 4):
        previous = values[-1]
        values.append(
            {
                s: sum(
                    softmax(actor[n][s])[a] * action_return(s, a, previous)
                    for a in ACTIONS
                )
                for s in STATES
            }
        )
    return {
        "values_by_remaining_actions": values,
        "average_of_n1_n2_n3_all_starts": sum(
            values[n][s] for n in (1, 2, 3) for s in STATES
        ) / 9,
    }


def actor_snapshot(actor: dict, critic: dict, optimal_actions: list, epoch: int) -> dict:
    exact = exact_policy_values(actor)
    choices = {
        n: {
            s: max(ACTIONS, key=lambda a: softmax(actor[n][s])[a])
            for s in STATES
        }
        for n in (1, 2, 3)
    }
    return {
        "epoch": epoch,
        "episodes_seen": epoch * 9,
        "exact_policy_objective_for_diagnostics": exact["average_of_n1_n2_n3_all_starts"],
        "exact_value_n3_energy2_for_diagnostics": exact["values_by_remaining_actions"][3][2],
        "greedy_action_matches_optimal_in_9_pairs": sum(
            choices[n][s] == optimal_actions[n][s]
            for n in (1, 2, 3) for s in STATES
        ),
        "p_n3_energy2": softmax(actor[3][2]),
        "critic_n3_energy2": critic[3][2],
    }


def train_actor_critic(epochs: int, seed: int, actor_rate: float, critic_rate: float) -> dict:
    actor, critic = blank_actor(), blank_critic()
    rng = random.Random(seed)
    exact_optimal = finite_model()["optimal_actions_by_remaining_actions"]
    checkpoints = [actor_snapshot(actor, critic, exact_optimal, 0)]
    for epoch in range(1, epochs + 1):
        starts = [(s, n) for n in (1, 2, 3) for s in STATES]
        rng.shuffle(starts)
        for energy, horizon in starts:
            events = sampled_actor_episode(actor, energy, horizon, rng)
            actor_critic_update(events, actor, critic, actor_rate, critic_rate)
        if epoch in CHECKPOINTS or epoch == epochs:
            checkpoints.append(actor_snapshot(actor, critic, exact_optimal, epoch))
    return {
        "epochs": epochs,
        "seed": seed,
        "actor_rate": actor_rate,
        "critic_rate": critic_rate,
        "sampling": "nine reset starts per epoch; current softmax policy draws actions; only observed events enter updates",
        "critic_target": "reward + current V[n-1][next state]; terminal V[0]=0",
        "actor_update": "old-episode action log probability score times the one-step TD residual",
        "checkpoints": checkpoints,
        "actor_logits": actor,
        "critic_values": critic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--actor-rate", type=float, default=0.03)
    parser.add_argument("--critic-rate", type=float, default=0.08)
    parser.add_argument(
        "--out", type=Path, default=Path("work/results/c23_policy_gradient.json")
    )
    args = parser.parse_args()
    if args.samples <= 1 or args.epochs <= 0:
        raise ValueError("samples must exceed one and epochs must be positive")
    if args.actor_rate <= 0 or args.critic_rate <= 0:
        raise ValueError("rates must be positive")
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"keep previous result; choose a new --out: {out}")

    branches = two_step_branches()
    exact = scalar_exact(0.0, branches)
    assert math.isclose(exact["objective"], 2.75, abs_tol=1e-12)
    assert math.isclose(exact["score_gradient"], -0.625, abs_tol=1e-12)
    assert math.isclose(
        exact["objective_baseline_estimator"]["variance"], 0.84375, abs_tol=1e-12
    )
    assert math.isclose(
        exact["zero_baseline_estimator"]["variance"], 2.734375, abs_tol=1e-12
    )
    on_policy = sample_two_step(0.5, 0.5, exact["objective"], args.samples, args.seed)
    off_exact = off_policy_exact(0.5, 0.8, branches)
    assert math.isclose(off_exact["uncorrected_expected_score"], 0.2, abs_tol=1e-12)
    assert math.isclose(
        off_exact["importance_corrected_expected_score"], -0.625, abs_tol=1e-12
    )
    off_asymmetric = off_policy_exact(0.3, 0.8, branches)
    assert math.isclose(
        off_asymmetric["uncorrected_centered_expected_score"], -1.025, abs_tol=1e-12
    )
    assert math.isclose(
        off_asymmetric["importance_corrected_centered_expected_score"], -0.525,
        abs_tol=1e-12,
    )
    off_sample = sample_two_step(0.5, 0.8, exact["objective"], args.samples, args.seed + 1)
    clipped = clipped_probe(0.5, 0.8, 0.2, branches)
    assert math.isclose(clipped["unclipped_surrogate"], -0.75, abs_tol=1e-12)
    assert math.isclose(clipped["clipped_surrogate"], -0.7875, abs_tol=1e-12)
    trained = train_actor_critic(args.epochs, args.seed + 2, args.actor_rate, args.critic_rate)
    output = {
        "scope": "C21 fictional fully observed courier; scalar two-step REINFORCE/baseline/off-policy/PPO ratio probes and finite-horizon tabular actor-critic; no neural PPO or language-model RL",
        "time_convention": "state_t -> action_t -> (reward_t+1, state_t+1)",
        "scalar_first_action": {
            "continuation": "after first action, follow C21 immediate policy for one final step",
            "exact_theta_zero": exact,
            "on_policy_sample": on_policy,
            "off_policy_exact": off_exact,
            "off_policy_asymmetric_check": off_asymmetric,
            "off_policy_sample": off_sample,
            "clipped_surrogate_probe": clipped,
            "all_equal_feedback_boundary": {
                "constant_return": -1.0,
                "expected_score_gradient": -1.0
                * (0.5 * score("express", 0.5) + 0.5 * score("slow", 0.5)),
                "meaning": "identical final feedback for every path carries no preference among the two first actions",
            },
        },
        "actor_critic": trained,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out)
    print("two-step J, exact gradient:", exact["objective"], exact["score_gradient"])
    print(
        "estimator variances, no baseline / J baseline:",
        exact["zero_baseline_estimator"]["variance"],
        exact["objective_baseline_estimator"]["variance"],
    )
    print(
        "off-policy exact naive / corrected:",
        off_exact["uncorrected_expected_score"],
        off_exact["importance_corrected_expected_score"],
    )
    print(
        "actor-critic exact start-average, first / last:",
        trained["checkpoints"][0]["exact_policy_objective_for_diagnostics"],
        trained["checkpoints"][-1]["exact_policy_objective_for_diagnostics"],
    )


if __name__ == "__main__":
    main()
