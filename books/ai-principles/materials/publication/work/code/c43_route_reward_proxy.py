"""A fixed C42 rollout showing what a progress-only score leaves out.

No policy is trained or optimized here. Both alternatives already existed in
the C42 simulation. This checks two evaluation rules at the same tick.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shapely.geometry import Point

from c42_bicycle_control import RUNS, Car, footprint, stress_polygon
from c42_commonroad_scene import load_scene


WORK = Path(__file__).resolve().parents[1]
SOURCE_REPORT = WORK / "runs/c42_control_verified/report.json"
SOURCE_TRACE = WORK / "runs/c42_control_verified/trace.json"
REPORT_SHA = "ae7aed1d3053f45a867ae941bd95800ed2ea23cc366b3119264f5b3579cd7f56"
TRACE_SHA = "7b42b26f8d277ab1a05763fb1d20affc427e2b1abf0c74d6b158fce0404b25c6"
TICK = 30


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out_path: Path):
    out = out_path.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new JSON file inside work/runs")
    if sha(SOURCE_REPORT) != REPORT_SHA or sha(SOURCE_TRACE) != TRACE_SHA:
        raise RuntimeError("C42 source rollout changed")
    scene = load_scene()
    trace = json.loads(SOURCE_TRACE.read_text(encoding="utf-8"))
    obstacle = stress_polygon(scene)
    first = Car(**trace["stress_no_brake"]["states"][0])
    start_s = scene.route.project(Point(first.x, first.y))
    rows = {}
    for name in ("stress_no_brake", "stress_with_brake"):
        states = trace[name]["states"]
        if len(states) <= TICK:
            raise RuntimeError(f"{name} stopped before tick {TICK}")
        car = Car(**states[TICK])
        progress = scene.route.project(Point(car.x, car.y)) - start_s
        collision = footprint(car).intersects(obstacle)
        rows[name] = {"route_progress_m": progress,
                      "author_stopped_car_overlap_at_tick": bool(collision),
                      "progress_only_score": progress,
                      "feasible_under_sampled_no_collision_rule": not collision}
    if not rows["stress_no_brake"]["author_stopped_car_overlap_at_tick"] or rows["stress_with_brake"]["author_stopped_car_overlap_at_tick"]:
        raise RuntimeError("the intended C42 collision contrast changed")
    progress_choice = max(rows, key=lambda name: rows[name]["progress_only_score"])
    feasible = [name for name in rows if rows[name]["feasible_under_sampled_no_collision_rule"]]
    constrained_choice = max(feasible, key=lambda name: rows[name]["route_progress_m"])
    report = {"c42_report_sha256": REPORT_SHA, "c42_trace_sha256": TRACE_SHA,
              "same_tick": TICK, "same_time_seconds": TICK * scene.dt,
              "conditions": "two author-written policies on one author kinematic vehicle and one author-created stopped car",
              "candidates": rows, "progress_only_prefers": progress_choice,
              "require_no_sampled_overlap_prefers": constrained_choice,
              "boundary": "The progress score omits collision and would prefer the collided candidate in this fixed pair; no RL agent was trained to exploit a reward, and discrete overlap cannot certify continuous safety."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "progress_only_prefers": progress_choice,
                      "require_no_sampled_overlap_prefers": constrained_choice}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out)
