"""One-road kinematic bicycle: replayed open loop vs measured closed loop.

The map is a pinned CommonRoad example. Other cars in its XML are SUMO
simulations and only available for their recorded 0.1 s time steps. An extra
stopped vehicle used for the braking stress is explicitly author-created.
This is a bounded numerical lesson, never a real-vehicle control system.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box

from c42_commonroad_scene import RUNS, load_scene

DT = 0.1
WHEELBASE = 2.8
REAR_TO_CENTER = 1.3
CAR_LENGTH = 4.5
CAR_WIDTH = 1.9
MAX_STEER = 0.52
MAX_STEER_RATE = 0.7
MAX_BRAKE = 4.0
MAX_ACCEL = 2.0
SPEED_TARGET = 6.0
STRESS_OBSTACLE_S = 46.0


def clip(x, low, high):
    return min(high, max(low, x))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Car:
    x: float  # rear axle, metres in scenario coordinates
    y: float
    yaw: float
    speed: float
    steer: float


def initial_car(scene):
    x, y, yaw, speed = scene.start  # XML position interpreted as car centre
    return Car(x - REAR_TO_CENTER * math.cos(yaw),
               y - REAR_TO_CENTER * math.sin(yaw), yaw, speed, 0.0)


def center(car):
    return (car.x + REAR_TO_CENTER * math.cos(car.yaw),
            car.y + REAR_TO_CENTER * math.sin(car.yaw))


def footprint(car):
    cx, cy = center(car)
    unit = box(-CAR_LENGTH / 2, -CAR_WIDTH / 2, CAR_LENGTH / 2, CAR_WIDTH / 2)
    return translate(rotate(unit, car.yaw, origin=(0, 0), use_radians=True), cx, cy)


def rectangle_at(x, y, yaw, length, width):
    unit = box(-length / 2, -width / 2, length / 2, width / 2)
    return translate(rotate(unit, yaw, origin=(0, 0), use_radians=True), x, y)


def pure_pursuit(car, scene):
    """Circle geometry: curvature = 2 sin(alpha) / chord length."""
    rear = Point(car.x, car.y)
    s = scene.route.project(rear)
    ahead_arc = max(6.0, 0.9 * car.speed)
    target = scene.route.interpolate(min(scene.route.length, s + ahead_arc))
    dx, dy = target.x - car.x, target.y - car.y
    chord = max(1e-6, math.hypot(dx, dy))
    alpha = wrap(math.atan2(dy, dx) - car.yaw)
    curvature = 2 * math.sin(alpha) / chord
    steer_cmd = clip(math.atan(WHEELBASE * curvature), -MAX_STEER, MAX_STEER)
    acceleration = clip(1.4 * (SPEED_TARGET - car.speed), -MAX_BRAKE, MAX_ACCEL)
    return acceleration, steer_cmd, s, target


def step(car, action, steering_bias=0.0):
    acceleration, desired_steer = action
    steer_delta = clip((desired_steer - car.steer) / DT,
                       -MAX_STEER_RATE, MAX_STEER_RATE) * DT
    steer = clip(car.steer + steer_delta, -MAX_STEER, MAX_STEER)
    actual_angle = clip(steer + steering_bias, -MAX_STEER, MAX_STEER)
    yaw_next = wrap(car.yaw + DT * car.speed * math.tan(actual_angle) / WHEELBASE)
    mid_yaw = wrap(car.yaw + wrap(yaw_next - car.yaw) / 2)
    x_next = car.x + DT * car.speed * math.cos(mid_yaw)
    y_next = car.y + DT * car.speed * math.sin(mid_yaw)
    speed_next = clip(car.speed + DT * clip(acceleration, -MAX_BRAKE, MAX_ACCEL), 0.0, 18.0)
    return Car(x_next, y_next, yaw_next, speed_next, steer)


def obstacle_at(scene, obstacle_id, tick):
    item = next(o for o in scene.obstacles if o.name == obstacle_id)
    if tick not in item.states:
        return None
    x, y, yaw, speed = item.states[tick]
    return rectangle_at(x, y, yaw, item.length, item.width), speed


def stress_polygon(scene):
    point = scene.route.interpolate(STRESS_OBSTACLE_S)
    nearby = scene.route.interpolate(min(scene.route.length, STRESS_OBSTACLE_S + 1))
    yaw = math.atan2(nearby.y - point.y, nearby.x - point.x)
    return rectangle_at(point.x, point.y, yaw, CAR_LENGTH, CAR_WIDTH)


def stop_action(car, action, scene, obstacle_s=STRESS_OBSTACLE_S):
    """Conservative 1D stopping check; static stress obstacle only."""
    s = scene.route.project(Point(car.x, car.y))
    center_gap = obstacle_s - s
    free = center_gap - (CAR_LENGTH + CAR_LENGTH) / 2 - 2.0
    # Stopping inequality v*reaction + v^2/(2b) <= free.
    needed = car.speed * 0.3 + car.speed**2 / (2 * MAX_BRAKE)
    safety_override = free <= needed
    acceleration = -MAX_BRAKE if safety_override else action[0]
    return (acceleration, action[1]), {"center_gap": center_gap, "free_gap": free,
                                       "needed": needed, "safety_override": safety_override,
                                       "commanded_acceleration": acceleration}


def simulate(scene, *, steps, bias=0.0, delay=0, actions=None, stress=False, braking=False):
    current = initial_car(scene)
    history = [current]
    used = []
    brake_log = []
    blocking_shape = stress_polygon(scene) if stress else None
    for tick in range(steps):
        if actions is None:
            observed = history[max(0, len(history) - 1 - delay)]
            acc, desired, _, _ = pure_pursuit(observed, scene)
            action = (acc, desired)
        else:
            action = actions[tick]
        if stress and braking:
            action, detail = stop_action(current, action, scene)
            brake_log.append(detail)
        used.append(action)
        current = step(current, action, steering_bias=bias)
        history.append(current)
        if blocking_shape is not None and footprint(current).intersects(blocking_shape):
            break  # teaching rollout ends at first geometric collision
    return history, used, brake_log


def metrics(scene, history, *, tick_limit=None, stress=False):
    horizon = history[:tick_limit + 1] if tick_limit is not None else history
    errors = []
    road = []
    goal = []
    progress = []
    min_traffic_gap = math.inf
    traffic_collisions = 0
    first_traffic_overlap = None
    stress_poly = None
    if stress:
        stress_poly = stress_polygon(scene)
    stress_gaps = []
    stress_collisions = 0
    first_stress_overlap = None
    for tick, car in enumerate(horizon):
        p = Point(car.x, car.y)
        poly = footprint(car)
        errors.append(scene.route.distance(p))
        road.append(scene.drivable.covers(poly))
        goal.append(scene.lanes[scene.goal_lane].polygon.covers(Point(center(car))))
        progress.append(scene.route.project(p))
        if stress_poly is not None:
            stress_gaps.append(poly.distance(stress_poly))
            if poly.intersects(stress_poly):
                stress_collisions += 1
                if first_stress_overlap is None:
                    first_stress_overlap = tick
        if tick_limit is not None:
            for obstacle in scene.obstacles:
                if tick not in obstacle.states:
                    continue
                x, y, yaw, _ = obstacle.states[tick]
                other = rectangle_at(x, y, yaw, obstacle.length, obstacle.width)
                min_traffic_gap = min(min_traffic_gap, poly.distance(other))
                if poly.intersects(other):
                    traffic_collisions += 1
                    if first_traffic_overlap is None:
                        first_traffic_overlap = tick
    return {"steps": len(horizon) - 1, "duration_s": (len(horizon) - 1) * DT,
            "end_progress_m": progress[-1] - progress[0],
            "end_route_s_m": progress[-1], "ever_goal_lane": any(goal),
            "mean_route_error_m": sum(errors) / len(errors), "max_route_error_m": max(errors),
            "offroad_steps": sum(not ok for ok in road),
            "traffic_min_shape_gap_m": None if tick_limit is None or math.isinf(min_traffic_gap) else min_traffic_gap,
            "traffic_collision_time_steps": traffic_collisions if tick_limit is not None else None,
            "first_original_traffic_overlap_step": first_traffic_overlap if tick_limit is not None else None,
            "synthetic_stop_min_shape_gap_m": min(stress_gaps) if stress_gaps else None,
            "synthetic_stop_collision_time_steps": stress_collisions if stress_poly else None,
            "first_synthetic_stop_overlap_step": first_stress_overlap if stress_poly else None,
            "end_speed_m_s": horizon[-1].speed}


def plot(scene, runs, out):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    choices = [(axes[0], ["nominal", "open_bias", "closed_bias", "closed_delay"], "同一路线：开环回放与闭环重读"),
               (axes[1], ["closed_bias", "stress_no_brake", "stress_with_brake"], "作者加停驶车：是否检查停车距离")]
    colors = {"nominal": "#444444", "open_bias": "#d16825", "closed_bias": "#1b7c74",
              "closed_delay": "#9859a5", "stress_no_brake": "#cf4a36", "stress_with_brake": "#166eac"}
    labels = {"nominal": "无偏差参考", "open_bias": "有偏差开环回放", "closed_bias": "有偏差逐步反馈",
              "closed_delay": "有偏差延迟反馈", "stress_no_brake": "不检查停驶车",
              "stress_with_brake": "检查停车距离"}
    for ax, names, title in choices:
        for lane in scene.lanes.values():
            for edge in (lane.left, lane.right):
                ax.plot(*zip(*edge), color="0.84", linewidth=0.45)
        ax.plot(*scene.route.xy, color="0.55", linewidth=1.7, label="目标车道中线")
        for name in names:
            cars = runs[name]["cars"]
            pts = [center(c) for c in cars]
            ax.plot(*zip(*pts), color=colors[name], linewidth=2, label=labels[name])
        ax.scatter(*center(runs["nominal"]["cars"][0]), s=30, color="black")
        if ax == axes[1]:
            stop = scene.route.interpolate(STRESS_OBSTACLE_S)
            nearby = scene.route.interpolate(min(scene.route.length, STRESS_OBSTACLE_S + 1))
            yaw = math.atan2(nearby.y - stop.y, nearby.x - stop.x)
            vehicle = rectangle_at(stop.x, stop.y, yaw, CAR_LENGTH, CAR_WIDTH)
            ax.fill(*vehicle.exterior.xy, color="#be2222", alpha=0.8, label="作者停驶车")
        ax.set_xlim(-135, -50)
        ax.set_ylim(-910, -840)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x（米）")
        ax.set_ylabel("y（米）")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / "control_paths.png", dpi=160)
    plt.close(fig)


def main(out_dir: Path, bias: float, delay: int, steps: int):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new work/runs output directory")
    scene = load_scene()
    if abs(scene.dt - DT) > 1e-12 or any(33 not in o.states for o in scene.obstacles):
        raise RuntimeError("expected 0.1 s scene and every original traffic trajectory through step 33")
    if not (0 <= bias <= 0.15 and 0 <= delay <= 10 and 33 <= steps <= 150):
        raise ValueError("outside bounded teaching variation")
    out.mkdir(parents=True)
    runs = {}
    nominal, nominal_actions, _ = simulate(scene, steps=steps)
    cases = {
        "nominal": (nominal, nominal_actions, []),
        "open_bias": simulate(scene, steps=steps, bias=bias, actions=nominal_actions),
        "closed_bias": simulate(scene, steps=steps, bias=bias),
        "closed_delay": simulate(scene, steps=steps, bias=bias, delay=delay),
        "stress_no_brake": simulate(scene, steps=steps, bias=bias, stress=True),
        "stress_with_brake": simulate(scene, steps=steps, bias=bias, stress=True, braking=True),
    }
    report = {"source_sha256": "2563a7dd4eedb60ef460d37afe4c0b718510092289f81341eb967bd3f741b8b4",
              "route_lanelets": scene.route_ids, "scenario_dt_s": scene.dt,
              "author_vehicle": {"wheelbase_m": WHEELBASE, "rear_to_center_m": REAR_TO_CENTER,
                                 "length_m": CAR_LENGTH, "width_m": CAR_WIDTH,
                                 "steering_rate_limit_rad_s": MAX_STEER_RATE,
                                 "steering_limit_rad": MAX_STEER, "speed_target_m_s": SPEED_TARGET},
              "teaching_conditions": {"steering_bias_rad": bias, "feedback_delay_steps": delay,
                                      "simulation_steps": steps, "original_other_traffic_last_shared_step": 33,
                                      "synthetic_stopped_obstacle_route_s_m": STRESS_OBSTACLE_S},
              "results": {}}
    for name, (cars, actions, brake_log) in cases.items():
        runs[name] = {"cars": cars, "actions": actions}
        report["results"][name] = {"road_only": metrics(scene, cars),
                                   "original_traffic_0_to_33": metrics(scene, cars, tick_limit=33),
                                   "synthetic_stopped_car_road_only": metrics(scene, cars, stress=True) if name.startswith("stress_") else None,
                                   "safety_override_steps": sum(x["safety_override"] for x in brake_log),
                                   "first_safety_override_tick": next((i for i, x in enumerate(brake_log) if x["safety_override"]), None),
                                   "first_safety_override_check": next((x for x in brake_log if x["safety_override"]), None)}
    plot(scene, runs, out)
    trace = {name: {"states": [vars(c) for c in run["cars"]],
                    "actions": [list(a) for a in run["actions"]]} for name, run in runs.items()}
    (out / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["trace"] = "trace.json"
    report["figure"] = "control_paths.png"
    report["boundary"] = "Author-defined kinematic bicycle and stopped-car stress on an OSM/SUMO CommonRoad example. Original other-car trajectories only used through step 33. Road-only beyond that is not a traffic safety result."
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"road_error": round(result["road_only"]["mean_route_error_m"], 3),
                             "offroad": result["road_only"]["offroad_steps"],
                             "stress_collision": result["synthetic_stopped_car_road_only"]["synthetic_stop_collision_time_steps"] if result["synthetic_stopped_car_road_only"] else None}
                      for name, result in report["results"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steering-bias", type=float, default=0.06)
    parser.add_argument("--feedback-delay", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    main(args.out_dir, args.steering_bias, args.feedback_delay, args.steps)
