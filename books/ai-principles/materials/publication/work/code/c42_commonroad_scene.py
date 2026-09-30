"""Read one fixed CommonRoad road/SUMO scenario without pretending it is a log.

The reader deliberately uses a small, inspectable subset of this pinned XML.
It is not a replacement for the general CommonRoad-IO implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
SOURCE = WORK / "data/c42_commonroad/ITA_Foggia-6_1_T-1.xml"
SOURCE_SHA = "2563a7dd4eedb60ef460d37afe4c0b718510092289f81341eb967bd3f741b8b4"


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xy(point):
    return float(point.findtext("x")), float(point.findtext("y"))


def bound(node):
    return [xy(point) for point in node.findall("point")]


def exact(node, name):
    return float(node.findtext(name + "/exact"))


@dataclass
class Lane:
    name: str
    left: list[tuple[float, float]]
    right: list[tuple[float, float]]
    successors: list[str]

    @property
    def polygon(self):
        return Polygon(self.left + list(reversed(self.right)))

    @property
    def middle(self):
        if len(self.left) != len(self.right):
            raise ValueError("selected route bounds need equal point counts")
        return [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(self.left, self.right)]


@dataclass
class Obstacle:
    name: str
    length: float
    width: float
    states: dict[int, tuple[float, float, float, float]]  # x,y,heading,speed


@dataclass
class Scene:
    lanes: dict[str, Lane]
    route_ids: list[str]
    route: LineString
    drivable: object
    start: tuple[float, float, float, float]
    goal_lane: str
    obstacles: list[Obstacle]
    dt: float


def load_scene() -> Scene:
    if sha(SOURCE) != SOURCE_SHA:
        raise RuntimeError("fixed CommonRoad source SHA changed")
    root = ET.parse(SOURCE).getroot()
    if root.tag != "commonRoad" or root.get("benchmarkID") != "ITA_Foggia-6_1_T-1":
        raise RuntimeError("unexpected scenario")
    lanes = {}
    graph = nx.DiGraph()
    for node in root.findall("lanelet"):
        name = node.get("id")
        successors = [x.get("ref") for x in node.findall("successor")]
        lane = Lane(name, bound(node.find("leftBound")), bound(node.find("rightBound")), successors)
        lanes[name] = lane
        graph.add_node(name)
        for nxt in successors:
            graph.add_edge(name, nxt)
    plan = root.find("planningProblem")
    start_node = plan.find("initialState")
    x, y = xy(start_node.find("position/point"))
    start = (x, y, exact(start_node, "orientation"), exact(start_node, "velocity"))
    goal = plan.find("goalState/position/lanelet").get("ref")
    here = Point(x, y)
    options = []
    for name, lane in lanes.items():
        if lane.polygon.covers(here):
            try:
                path = nx.shortest_path(graph, name, goal)
                options.append(path)
            except nx.NetworkXNoPath:
                pass
    if len(options) != 1:
        raise RuntimeError(f"route ambiguous: {options}")
    route_ids = options[0]
    centers = []
    for name in route_ids:
        for point in lanes[name].middle:
            if not centers or math.dist(point, centers[-1]) > 1e-6:
                centers.append(point)
    route = LineString(centers)
    road = unary_union([lane.polygon.buffer(0) for lane in lanes.values()])
    obstacles = []
    for node in root.findall("dynamicObstacle"):
        rectangle = node.find("shape/rectangle")
        length, width = float(rectangle.findtext("length")), float(rectangle.findtext("width"))
        states = {}
        for state in [node.find("initialState"), *node.findall("trajectory/state")]:
            step = int(exact(state, "time"))
            px, py = xy(state.find("position/point"))
            states[step] = (px, py, exact(state, "orientation"), exact(state, "velocity"))
        obstacles.append(Obstacle(node.get("id"), length, width, states))
    return Scene(lanes, route_ids, route, road, start, goal, obstacles, float(root.get("timeStepSize")))


def inspect(out_dir: Path):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new work/runs directory")
    scene = load_scene()
    out.mkdir(parents=True)
    projection = scene.route.project(Point(scene.start[:2]))
    nearby = []
    for item in scene.obstacles:
        x, y, heading, speed = item.states[0]
        point = Point(x, y)
        nearby.append({"id": item.name, "speed_m_s": speed,
                       "distance_to_route_m": scene.route.distance(point),
                       "route_position_m": scene.route.project(point),
                       "initial_xy": [x, y], "available_steps": len(item.states)})
    nearby.sort(key=lambda x: x["distance_to_route_m"])
    fig, ax = plt.subplots(figsize=(11, 8))
    for lane in scene.lanes.values():
        for edge in (lane.left, lane.right):
            ax.plot(*zip(*edge), color="0.75", linewidth=0.7)
    ax.plot(*scene.route.xy, color="#c83d3d", linewidth=2.2, label="所选车道中心路线")
    ax.scatter([scene.start[0]], [scene.start[1]], color="#108273", s=75, label="自车初态")
    for item in scene.obstacles:
        x, y, *_ = item.states[0]
        ax.scatter([x], [y], color="#2966af", s=16)
    ax.set_xlim(scene.start[0] - 120, scene.start[0] + 90)
    ax.set_ylim(scene.start[1] - 100, scene.start[1] + 95)
    ax.set_aspect("equal")
    ax.set_xlabel("场景平面 x（米）")
    ax.set_ylabel("场景平面 y（米）")
    ax.set_title("CommonRoad Foggia：OSM道路 / SUMO车辆 / 作者选路线")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.tight_layout()
    fig.savefig(out / "road_and_route.png", dpi=150)
    plt.close(fig)
    report = {"source": str(SOURCE.relative_to(WORK)), "source_sha256": SOURCE_SHA,
              "scenario_id": "ITA_Foggia-6_1_T-1", "dt_seconds": scene.dt,
              "road_source": "OpenStreetMap road geometry; SUMO simulated traffic, as XML header states",
              "lanelets": len(scene.lanes), "route_lanelets": scene.route_ids,
              "route_length_m": scene.route.length, "start_route_projection_m": projection,
              "start_xy_heading_speed": list(scene.start), "goal_lanelet": scene.goal_lane,
              "dynamic_obstacles": len(scene.obstacles), "obstacles_by_initial_route_distance": nearby,
              "figure": "road_and_route.png"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"route": scene.route_ids, "length_m": scene.route.length,
                      "start_s": projection, "nearest_obstacles": nearby[:3]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.out_dir)
