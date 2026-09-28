"""C24: train a small DQN on Gymnasium CartPole's four numeric observations.

This is not the pixel Atari experiment of Mnih et al. The network, replay,
target network, reward source, and time-limit semantics are explicit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


def run_policy(model: QNetwork | None, seeds: list[int], *, random_seed: int = 0) -> list[int]:
    env = gym.make("CartPole-v1")
    rng = random.Random(random_seed)
    lengths = []
    try:
        for seed in seeds:
            state, _ = env.reset(seed=seed)
            length = 0
            while True:
                if model is None:
                    action = rng.randrange(2)
                else:
                    with torch.no_grad():
                        q = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                        action = int(q.argmax(dim=1).item())
                state, _, terminated, truncated, _ = env.step(action)
                length += 1
                if terminated or truncated:
                    lengths.append(length)
                    break
    finally:
        env.close()
    return lengths


def optimize(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: deque,
    batch_size: int,
    gamma: float,
    rng: random.Random,
) -> float:
    batch = rng.sample(replay, batch_size)
    states = torch.from_numpy(np.stack([row[0] for row in batch]))
    actions = torch.tensor([row[1] for row in batch], dtype=torch.int64)
    rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
    next_states = torch.from_numpy(np.stack([row[3] for row in batch]))
    terminated = torch.tensor([row[4] for row in batch], dtype=torch.float32)
    chosen_q = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_best = target(next_states).max(dim=1).values
        desired = rewards + gamma * (1.0 - terminated) * next_best
    loss = F.smooth_l1_loss(chosen_q, desired)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--target-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--out-report", type=Path, default=Path("work/results/c24_cartpole_dqn_train.json"))
    parser.add_argument("--out-model", type=Path, default=Path("work/results/c24_cartpole_dqn_best.pt"))
    args = parser.parse_args()
    if min(args.episodes, args.batch_size, args.warmup, args.target_every, args.eval_every) <= 0:
        raise ValueError("all counts must be positive")
    if args.warmup < args.batch_size:
        raise ValueError("warmup must cover at least one batch")
    report_path = args.out_report if args.out_report.is_absolute() else ROOT / args.out_report
    model_path = args.out_model if args.out_model.is_absolute() else ROOT / args.out_model
    if report_path.exists() or model_path.exists():
        raise FileExistsError("keep old run; choose new report and model paths")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    online = QNetwork()
    target = QNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
    replay: deque = deque(maxlen=50_000)
    env = gym.make("CartPole-v1")
    validation_seeds = list(range(10_000, 10_020))
    initial_validation = run_policy(online, validation_seeds)
    random_validation = run_policy(None, validation_seeds, random_seed=args.seed + 100)
    best_mean = statistics.fmean(initial_validation)
    best_episode = 0
    best_state = copy.deepcopy(online.state_dict())
    training_lengths = []
    updates = 0
    transitions = 0
    physical_terminations = 0
    time_limit_truncations = 0
    recent_losses: list[float] = []
    records = []
    try:
        for episode in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + episode)
            length = 0
            while True:
                epsilon = max(0.05, 1.0 - 0.95 * min(transitions / 10_000, 1.0))
                if rng.random() < epsilon:
                    action = rng.randrange(2)
                else:
                    with torch.no_grad():
                        q = online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                        action = int(q.argmax(dim=1).item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                replay.append((
                    np.asarray(state, dtype=np.float32).copy(),
                    action,
                    float(reward),
                    np.asarray(next_state, dtype=np.float32).copy(),
                    bool(terminated),
                ))
                transitions += 1
                length += 1
                if len(replay) >= args.warmup:
                    recent_losses.append(
                        optimize(online, target, optimizer, replay, args.batch_size, 0.99, rng)
                    )
                    updates += 1
                    if updates % args.target_every == 0:
                        target.load_state_dict(online.state_dict())
                state = next_state
                if terminated or truncated:
                    physical_terminations += int(terminated)
                    time_limit_truncations += int(truncated)
                    training_lengths.append(length)
                    break

            if episode % args.eval_every == 0 or episode == args.episodes:
                validation = run_policy(online, validation_seeds)
                validation_mean = statistics.fmean(validation)
                if validation_mean > best_mean:
                    best_mean = validation_mean
                    best_episode = episode
                    best_state = copy.deepcopy(online.state_dict())
                records.append({
                    "after_episode": episode,
                    "training_last_window_mean": statistics.fmean(
                        training_lengths[-args.eval_every :]
                    ),
                    "validation_lengths": validation,
                    "validation_mean": validation_mean,
                    "epsilon_at_end": epsilon,
                    "transitions": transitions,
                    "optimizer_updates": updates,
                    "replay_items": len(replay),
                    "mean_loss_since_last_validation": (
                        statistics.fmean(recent_losses) if recent_losses else None
                    ),
                })
                recent_losses.clear()
                print(
                    f"episode {episode}: train-last {records[-1]['training_last_window_mean']:.1f}, "
                    f"validation {validation_mean:.1f}, best {best_mean:.1f}, "
                    f"steps {transitions}, updates {updates}",
                    flush=True,
                )
    finally:
        env.close()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model = model_path.with_name(model_path.name + ".tmp")
    torch.save(
        {
            "state_dict": best_state,
            "architecture": "4-128-128-2 ReLU",
            "selected_after_episode": best_episode,
            "validation_mean": best_mean,
            "seed": args.seed,
        },
        temp_model,
    )
    os.replace(temp_model, model_path)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report = {
        "scope": "Gymnasium CartPole-v1 four numeric observations; model-free DQN-style local training with replay and a separate target network; not pixel Atari or original DQN compute",
        "gymnasium_version": gym.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
        "seed": args.seed,
        "environment": {
            "observation": "cart position, cart velocity, pole angle, pole angular velocity",
            "actions": "0 left; 1 right",
            "reward": "Gymnasium v1.3.0 CartPole-v1 default +1 each step including physical termination step",
            "episode_end": "physical termination or 500-step TimeLimit truncation",
            "bootstrap": "target zeroes future only for physical terminated, not TimeLimit truncated",
        },
        "configuration": {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "warmup_items": args.warmup,
            "target_copy_every_updates": args.target_every,
            "replay_capacity": 50_000,
            "gamma": 0.99,
            "epsilon": "linear from 1.0 to 0.05 over first 10000 transitions",
            "optimizer": "Adam lr=0.001; smooth_l1_loss; grad norm clip=10",
            "network": "4-128-128-2 ReLU MLP; new random weights",
        },
        "validation_seeds": validation_seeds,
        "initial_validation_lengths": initial_validation,
        "initial_validation_mean": statistics.fmean(initial_validation),
        "random_validation_lengths": random_validation,
        "random_validation_mean": statistics.fmean(random_validation),
        "training_lengths": training_lengths,
        "records": records,
        "physical_terminations": physical_terminations,
        "time_limit_truncations": time_limit_truncations,
        "best_checkpoint": {
            "path": str(model_path.relative_to(ROOT)),
            "sha256": model_sha,
            "selected_after_episode": best_episode,
            "validation_mean": best_mean,
        },
        "target_convention_probe": {
            "reward": 1.0,
            "hypothetical_next_best_q": 10.0,
            "if_physical_terminated": 1.0,
            "if_only_time_limit_truncated": 1.0 + 0.99 * 10.0,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_report = report_path.with_name(report_path.name + ".tmp")
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_report, report_path)
    print("best validation", best_mean, "episode", best_episode)
    print("model sha256", model_sha)


if __name__ == "__main__":
    main()
