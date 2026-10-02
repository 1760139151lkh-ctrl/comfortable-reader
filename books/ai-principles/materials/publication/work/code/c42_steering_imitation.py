"""Imitate this book's pure-pursuit teacher on one fixed road, then roll out.

Teacher labels are author-computed steering commands, not human driving logs.
The learner sees route-relative numerical features, never camera pixels.
Train/validation are used before opening two predeclared test condition sets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Point
from torch import nn

from c42_bicycle_control import (DT, MAX_ACCEL, MAX_BRAKE, RUNS, Car, center,
                                 clip, footprint, initial_car, metrics, pure_pursuit,
                                 step, wrap)
from c42_commonroad_scene import load_scene

WORK = Path(__file__).resolve().parents[1]
STEPS = 80
TRAIN_SEED = 20260925
VAL_SEED = 31415
TEST_SEED = 27182
CHALLENGE_SEED = 16180
EPOCHS = 80


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions(scene, count, seed, wider=False):
    rng = random.Random(seed)
    result = []
    attempts = 0
    while len(result) < count and attempts < 10000:
        attempts += 1
        lateral = rng.uniform(-0.95 if wider else -0.45, 0.95 if wider else 0.45)
        heading = rng.uniform(-0.14 if wider else -0.055, 0.14 if wider else 0.055)
        bias = rng.uniform(-0.09 if wider else -0.04, 0.09 if wider else 0.04)
        base = initial_car(scene)
        car = Car(base.x - math.sin(base.yaw) * lateral,
                  base.y + math.cos(base.yaw) * lateral,
                  wrap(base.yaw + heading), base.speed, 0.0)
        if scene.drivable.covers(footprint(car)):
            result.append({"lateral_m": lateral, "heading_rad": heading,
                           "bias_rad": bias})
    if len(result) != count:
        raise RuntimeError("could not sample initial states on the road")
    return result


def start_for(scene, condition):
    base = initial_car(scene)
    lateral = condition["lateral_m"]
    return Car(base.x - math.sin(base.yaw) * lateral,
               base.y + math.cos(base.yaw) * lateral,
               wrap(base.yaw + condition["heading_rad"]), base.speed, 0.0)


def path_heading(route, s):
    first = route.interpolate(max(0.0, s - 0.4))
    second = route.interpolate(min(route.length, s + 0.4))
    return math.atan2(second.y - first.y, second.x - first.x)


def features(car, scene):
    s = scene.route.project(Point(car.x, car.y))
    near = scene.route.interpolate(s)
    yaw = path_heading(scene.route, s)
    left_x, left_y = -math.sin(yaw), math.cos(yaw)
    lateral = ((car.x - near.x) * left_x + (car.y - near.y) * left_y)
    future_heading = path_heading(scene.route, min(scene.route.length - 0.5, s + 8.0))
    curvature_hint = wrap(future_heading - yaw) / 8.0
    return np.array([lateral / 2, wrap(yaw - car.yaw), car.speed / 10,
                     car.steer, curvature_hint * 10, (scene.route.length - s) / scene.route.length],
                    dtype=np.float32)


class SteeringNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


def rollout(scene, condition, model=None, normalizer=None):
    car = start_for(scene, condition)
    states = [car]
    pairs = []
    for _ in range(STEPS):
        acc, expert_steer, _, _ = pure_pursuit(car, scene)
        observation = features(car, scene)
        pairs.append((observation, expert_steer))
        if model is None:
            command = expert_steer
        else:
            mean, std = normalizer
            with torch.no_grad():
                inputs = torch.from_numpy((observation - mean) / std).unsqueeze(0)
                command = float(model(inputs).item())
            command = clip(command, -0.52, 0.52)
        # Longitudinal control remains the explicit teacher's speed rule.
        car = step(car, (acc, command), steering_bias=condition["bias_rad"])
        states.append(car)
    return states, pairs


def examples(scene, group, policy=None, normalizer=None):
    states_and_pairs = [rollout(scene, condition, policy, normalizer) for condition in group]
    xs = np.stack([feature for _, pairs in states_and_pairs for feature, _ in pairs])
    ys = np.array([label for _, pairs in states_and_pairs for _, label in pairs], dtype=np.float32)
    return xs, ys, states_and_pairs


def train_model(xs, ys, val_xs, val_ys, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    net = SteeringNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    mean = xs.mean(axis=0)
    std = np.maximum(xs.std(axis=0), 1e-3)
    tx = torch.from_numpy((xs - mean) / std)
    ty = torch.from_numpy(ys)
    vx = torch.from_numpy((val_xs - mean) / std)
    vy = torch.from_numpy(val_ys)
    generator = torch.Generator().manual_seed(seed)
    best = None
    best_val = math.inf
    best_epoch = None
    curve = []
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for ids in torch.randperm(len(tx), generator=generator).split(128):
            loss = (net(tx[ids]) - ty[ids]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            training_loss = float((net(tx) - ty).square().mean().item())
            validation_loss = float((net(vx) - vy).square().mean().item())
        curve.append({"epoch": epoch, "train_mse": training_loss, "validation_mse": validation_loss})
        if validation_loss < best_val:
            best_val, best_epoch = validation_loss, epoch
            best = copy.deepcopy(net.state_dict())
    net.load_state_dict(best)
    return net, (mean, std), {"best_epoch": best_epoch, "best_validation_mse": best_val,
                              "curve": curve, "training_labels": len(ys), "validation_labels": len(val_ys)}


def group_score(scene, group, model, normalizer):
    rows = []
    for condition in group:
        states, _ = rollout(scene, condition, model, normalizer)
        result = metrics(scene, states)
        rows.append({"mean_route_error_m": result["mean_route_error_m"],
                     "max_route_error_m": result["max_route_error_m"],
                     "offroad_steps": result["offroad_steps"],
                     "ever_goal_lane": result["ever_goal_lane"]})
    return {"episodes": len(rows), "mean_episode_route_error_m": sum(x["mean_route_error_m"] for x in rows) / len(rows),
            "episodes_with_offroad": sum(x["offroad_steps"] > 0 for x in rows),
            "total_offroad_time_steps": sum(x["offroad_steps"] for x in rows),
            "episodes_entering_goal_lane": sum(x["ever_goal_lane"] for x in rows),
            "rows": rows}


def save_model(path, model, normalizer):
    mean, std = normalizer
    torch.save({"model": model.state_dict(), "feature_mean": torch.from_numpy(mean.copy()),
                "feature_std": torch.from_numpy(std.copy()),
                "source_sha256": "2563a7dd4eedb60ef460d37afe4c0b718510092289f81341eb967bd3f741b8b4"}, path)


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = SteeringNet()
    model.load_state_dict(state["model"])
    model.eval()
    return model, (state["feature_mean"].numpy(), state["feature_std"].numpy())


def train(out_dir):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new work/runs directory")
    scene = load_scene()
    out.mkdir(parents=True)
    training = conditions(scene, 40, TRAIN_SEED)
    validation = conditions(scene, 10, VAL_SEED)
    train_x, train_y, _ = examples(scene, training)
    val_x, val_y, _ = examples(scene, validation)
    bc, bc_norm, bc_curve = train_model(train_x, train_y, val_x, val_y, seed=42)
    # One explicit DAgger-style aggregation: collect states visited by the BC learner
    # on training conditions, then ask the same deterministic teacher for labels.
    learner_x, learner_y, _ = examples(scene, training, bc, bc_norm)
    dagger_x = np.concatenate([train_x, learner_x])
    dagger_y = np.concatenate([train_y, learner_y])
    dagger, dagger_norm, dagger_curve = train_model(dagger_x, dagger_y, val_x, val_y, seed=42)
    save_model(out / "bc.pt", bc, bc_norm)
    save_model(out / "dagger_one_round.pt", dagger, dagger_norm)
    bc_val = group_score(scene, validation, bc, bc_norm)
    dagger_val = group_score(scene, validation, dagger, dagger_norm)
    key = lambda score: (score["episodes_with_offroad"], score["mean_episode_route_error_m"])
    selected = "dagger_one_round" if key(dagger_val) < key(bc_val) else "bc"
    report = {"source_xml_sha256": "2563a7dd4eedb60ef460d37afe4c0b718510092289f81341eb967bd3f741b8b4",
              "script_sha256": sha(Path(__file__)), "teacher": "author's geometry pure pursuit; longitudinal speed rule remains explicit",
              "feature_names": ["signed_lateral_error_over_2", "path_heading_minus_yaw", "speed_over_10",
                                "current_steer", "ahead_curvature_times_10", "remaining_fraction"],
              "split": {"train_episodes": len(training), "validation_episodes": len(validation),
                        "train_seed": TRAIN_SEED, "validation_seed": VAL_SEED,
                        "test_seed_unopened": TEST_SEED, "challenge_seed_unopened": CHALLENGE_SEED,
                        "train_conditions": training, "validation_conditions": validation},
              "bc": {"training": bc_curve, "validation_rollout": bc_val, "model_sha256": sha(out / "bc.pt")},
              "dagger_one_round": {"training": dagger_curve, "validation_rollout": dagger_val,
                                   "extra_teacher_labels_on_bc_states": len(learner_y),
                                   "model_sha256": sha(out / "dagger_one_round.pt")},
              "selection_rule": "fewest validation episodes with offroad; then lowest mean validation episode route error",
              "selected_before_test": selected,
              "scope": "same authored map, authored vehicle dynamics and authored teacher; no real driver, image perception or independent road distribution"}
    (out / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bc_val_error": bc_val["mean_episode_route_error_m"],
                      "dagger_val_error": dagger_val["mean_episode_route_error_m"],
                      "bc_val_offroad": bc_val["episodes_with_offroad"],
                      "dagger_val_offroad": dagger_val["episodes_with_offroad"],
                      "selected_before_test": selected}, ensure_ascii=False))


def test(run_dir):
    out = run_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or (out / "test.json").exists():
        raise ValueError("run directory must be inside work/runs, test unopened")
    train_report = json.loads((out / "train.json").read_text(encoding="utf-8"))
    if train_report["script_sha256"] != sha(Path(__file__)):
        raise RuntimeError("training code changed; review before first test")
    for name, file in (("bc", "bc.pt"), ("dagger_one_round", "dagger_one_round.pt")):
        if sha(out / file) != train_report[name]["model_sha256"]:
            raise RuntimeError("selected model file changed")
    scene = load_scene()
    test_groups = {"same_distribution_new_episodes": conditions(scene, 10, TEST_SEED),
                   "wider_initial_and_bias": conditions(scene, 10, CHALLENGE_SEED, wider=True)}
    models = {name: load_model(out / ("bc.pt" if name == "bc" else "dagger_one_round.pt"))
              for name in ("bc", "dagger_one_round")}
    results = {}
    for group_name, group in test_groups.items():
        teacher = group_score(scene, group, None, None)
        results[group_name] = {"teacher": teacher}
        for name, (model, norm) in models.items():
            results[group_name][name] = group_score(scene, group, model, norm)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 6))
    for lane in scene.lanes.values():
        for edge in (lane.left, lane.right):
            ax.plot(*zip(*edge), color="0.82", linewidth=0.5)
    first = test_groups["wider_initial_and_bias"][0]
    for name, params, color in (("教师几何控制", None, "#333333"),
                                ("只用专家轨迹模仿", models["bc"], "#d26d27"),
                                ("加一轮学生状态标注", models["dagger_one_round"], "#168277")):
        states, _ = rollout(scene, first, *(params or (None, None)))
        positions = [center(s) for s in states]
        ax.plot(*zip(*positions), label=name, color=color, linewidth=2)
    ax.set_xlim(-135, -50)
    ax.set_ylim(-910, -840)
    ax.set_aspect("equal")
    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")
    ax.set_title("同一路线、未见初态：教师与两种数值状态模仿")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / "imitation_test_paths.png", dpi=150)
    plt.close(fig)
    report = {"training_report_sha256": sha(out / "train.json"),
              "selected_before_test": train_report["selected_before_test"],
              "test_episode_seeds": {"same_distribution": TEST_SEED, "wider": CHALLENGE_SEED},
              "tests_first_opened_together": True, "results": results,
              "figure": "imitation_test_paths.png",
              "scope": "both branches reported after validation decision; not a second road, human steering or real vehicle safety study"}
    (out / "test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {group: {name: {"error": round(x["mean_episode_route_error_m"], 4),
                               "offroad_episodes": x["episodes_with_offroad"]}
                       for name, x in values.items()} for group, values in results.items()}
    print(json.dumps({"selected_before_test": train_report["selected_before_test"],
                      "results": summary}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("train").add_argument("--run-dir", type=Path, required=True)
    sub.add_parser("test").add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "train":
        train(args.run_dir)
    else:
        test(args.run_dir)
