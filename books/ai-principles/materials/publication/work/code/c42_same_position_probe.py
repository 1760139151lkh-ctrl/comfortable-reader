"""At one road location, heading changes the next action and displacement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from c42_bicycle_control import RUNS, Car, initial_car, pure_pursuit, step, wrap
from c42_commonroad_scene import load_scene


def main(out_path: Path):
    out = out_path.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new work/runs JSON path")
    scene = load_scene()
    base = initial_car(scene)
    rows = []
    for shift in (-0.15, 0.0, 0.15):
        car = Car(base.x, base.y, wrap(base.yaw + shift), base.speed, base.steer)
        acceleration, steering, s, target = pure_pursuit(car, scene)
        after = step(car, (acceleration, steering))
        rows.append({"heading_shift_rad": shift, "same_rear_xy": [car.x, car.y],
                     "same_speed_m_s": car.speed, "target_xy": [target.x, target.y],
                     "steer_command_rad": steering, "acceleration_m_s2": acceleration,
                     "after_one_tick_rear_xy": [after.x, after.y],
                     "route_projection_m": s})
    report = {"source": "ITA_Foggia-6_1_T-1.xml fixed route plus author kinematic bicycle",
              "same_visible_position_different_heading": rows,
              "claim_boundary": "same position and speed do not determine a unique heading or steering action; this is a deterministic mechanism probe, not learned state estimation"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"steer_commands": [x["steer_command_rad"] for x in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out)
