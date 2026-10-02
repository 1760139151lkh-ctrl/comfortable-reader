"""C11：同一把钥匙、门和目标下的 BFS、A* 与有限域约束求解。"""

import heapq
import itertools
import json
import platform
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ortools
from ortools.sat.python import cp_model


WORK = Path(__file__).resolve().parents[1]
GRID = (
    "#########",
    "#S.K#...#",
    "#.###...#",
    "#...#...#",
    "#...#...#",
    "#...D..G#",
    "#########",
)
ACTIONS = (
    ("N", (0, -1)),
    ("E", (1, 0)),
    ("S", (0, 1)),
    ("W", (-1, 0)),
)
MAX_HORIZON = 24
SOLVER_SECONDS_PER_HORIZON = 10.0


def parse_grid():
    if len(GRID) != 7 or any(len(row) != 9 for row in GRID):
        raise ValueError("地图尺寸与协议不同")
    places = {letter: [] for letter in "SKDG"}
    walkable = set()
    for y, row in enumerate(GRID):
        for x, ch in enumerate(row):
            if ch not in "#.SKDG":
                raise ValueError(f"未定义地图字符 {ch}")
            if ch != "#":
                walkable.add((x, y))
            if ch in places:
                places[ch].append((x, y))
    for letter in ("S", "K", "D", "G"):
        if len(places[letter]) != 1:
            raise ValueError(f"{letter} 必须恰有一处")
    return walkable, {name: spots[0] for name, spots in places.items()}


WALKABLE, SPECIAL = parse_grid()
START = (*SPECIAL["S"], False)
GOAL = (*SPECIAL["G"], True)


def next_state(state, action):
    x, y, has_key = state
    dx, dy = dict(ACTIONS)[action]
    target = (x + dx, y + dy)
    if target not in WALKABLE:
        return None
    if target == SPECIAL["D"] and not has_key:
        return None
    new_key = has_key or target == SPECIAL["K"]
    return (*target, new_key)


def successors(state):
    for action, _ in ACTIONS:
        reached = next_state(state, action)
        if reached is not None:
            yield action, reached


def rebuild(parent, end):
    actions = []
    states = [end]
    state = end
    while parent[state] is not None:
        old, action = parent[state]
        actions.append(action)
        state = old
        states.append(state)
    actions.reverse()
    states.reverse()
    return actions, states


def bfs(only_position=False):
    queue = deque([START])
    parent = {START: None}
    visited = {START[:2] if only_position else START}
    expanded = 0
    peak = 1
    while queue:
        state = queue.popleft()
        expanded += 1
        if state == GOAL:
            actions, states = rebuild(parent, state)
            return {
                "actions": actions, "states": states,
                "expanded": expanded, "discovered": len(visited),
                "peak_frontier": peak,
            }
        for action, neighbor in successors(state):
            identity = neighbor[:2] if only_position else neighbor
            if identity in visited:
                continue
            visited.add(identity)
            parent[neighbor] = (state, action)
            queue.append(neighbor)
        peak = max(peak, len(queue))
    return {
        "actions": None, "states": None,
        "expanded": expanded, "discovered": len(visited),
        "peak_frontier": peak,
    }


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def goal_distance(state):
    return manhattan(state, SPECIAL["G"])


def key_aware_distance(state):
    if state[2]:
        return manhattan(state, SPECIAL["G"])
    return manhattan(state, SPECIAL["K"]) + manhattan(SPECIAL["K"], SPECIAL["G"])


def astar(heuristic):
    order = itertools.count()
    frontier = [(heuristic(START), heuristic(START), next(order), START)]
    best_g = {START: 0}
    expanded_at = {}
    parent = {START: None}
    expanded = 0
    peak = 1
    while frontier:
        f, h, _, state = heapq.heappop(frontier)
        g = f - h
        if g != best_g.get(state) or expanded_at.get(state, float("inf")) <= g:
            continue
        expanded_at[state] = g
        expanded += 1
        if state == GOAL:
            actions, states = rebuild(parent, state)
            return {
                "actions": actions, "states": states,
                "expanded": expanded, "discovered": len(best_g),
                "peak_frontier": peak,
            }
        for action, neighbor in successors(state):
            new_g = g + 1
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                parent[neighbor] = (state, action)
                new_h = heuristic(neighbor)
                heapq.heappush(frontier,
                               (new_g + new_h, new_h, next(order), neighbor))
        peak = max(peak, len(frontier))
    raise RuntimeError("A* 未到达目标")


def all_states():
    return [(x, y, has_key)
            for y in range(len(GRID))
            for x in range(len(GRID[0]))
            if (x, y) in WALKABLE
            for has_key in (False, True)]


def true_remaining_distances(states):
    reverse = {state: [] for state in states}
    for state in states:
        for _, neighbor in successors(state):
            reverse[neighbor].append(state)
    distances = {GOAL: 0}
    queue = deque([GOAL])
    while queue:
        state = queue.popleft()
        for earlier in reverse[state]:
            if earlier not in distances:
                distances[earlier] = distances[state] + 1
                queue.append(earlier)
    return distances


def verify_heuristics(states):
    true_distance = true_remaining_distances(states)
    report = {}
    for name, heuristic in (
        ("goal_manhattan", goal_distance),
        ("key_aware", key_aware_distance),
    ):
        if heuristic(GOAL) != 0:
            raise ValueError(f"{name} 对目标未报零")
        transitions = 0
        for state in states:
            if state in true_distance and heuristic(state) > true_distance[state]:
                raise ValueError(f"{name} 高估状态 {state} 的剩余成本")
            for _, neighbor in successors(state):
                transitions += 1
                if heuristic(state) > 1 + heuristic(neighbor):
                    raise ValueError(f"{name} 在 {state}->{neighbor} 不一致")
        report[name] = {
            "checked_states_with_finite_goal_distance": len(true_distance),
            "checked_legal_transitions": transitions,
            "start_lower_bound": heuristic(START),
            "true_start_distance": true_distance[START],
        }
    return report


def replay(actions):
    state = START
    path = [state]
    for step, action in enumerate(actions, start=1):
        if action not in dict(ACTIONS):
            raise ValueError(f"第 {step} 步不是允许行动")
        state = next_state(state, action)
        if state is None:
            raise ValueError(f"第 {step} 步穿墙或无钥匙进门")
        path.append(state)
    if state != GOAL:
        raise ValueError("计划走完后未到带钥匙的目标状态")
    return path


def constraint_plan():
    states = all_states()
    state_id = {state: i for i, state in enumerate(states)}
    tuples = []
    for state in states:
        for action_id, (action, _) in enumerate(ACTIONS):
            neighbor = next_state(state, action)
            if neighbor is not None:
                tuples.append((state_id[state], action_id, state_id[neighbor]))
    statuses = []
    for horizon in range(MAX_HORIZON + 1):
        model = cp_model.CpModel()
        s = [model.new_int_var(0, len(states) - 1, f"s_{t}")
             for t in range(horizon + 1)]
        a = [model.new_int_var(0, len(ACTIONS) - 1, f"a_{t}")
             for t in range(horizon)]
        model.add(s[0] == state_id[START])
        model.add(s[-1] == state_id[GOAL])
        for t in range(horizon):
            model.add_allowed_assignments([s[t], a[t], s[t + 1]], tuples)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 20260924
        solver.parameters.max_time_in_seconds = SOLVER_SECONDS_PER_HORIZON
        status = solver.solve(model)
        status_name = solver.status_name(status)
        statuses.append({"horizon": horizon, "status": status_name})
        if status == cp_model.INFEASIBLE:
            continue
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"T={horizon} 未获确定答案：{status_name}")
        actions = [ACTIONS[solver.value(var)][0] for var in a]
        return {
            "actions": actions,
            "states": replay(actions),
            "first_feasible_horizon": horizon,
            "statuses": statuses,
            "allowed_transition_tuples": len(tuples),
            "finite_state_count": len(states),
            "solver": f"OR-Tools CP-SAT {ortools.__version__}",
        }
    raise RuntimeError(f"到 {MAX_HORIZON} 步仍无计划")


def draw_figure(plan, methods):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for y, row in enumerate(GRID):
        for x, ch in enumerate(row):
            colors = {
                "#": "#30343b", ".": "#f3f5f6", "S": "#cfe3ee",
                "K": "#ffe3a6", "D": "#d9b2a5", "G": "#c6e5c0",
            }
            ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1, facecolor=colors[ch],
                edgecolor="#bac3ca", linewidth=0.8
            ))
            if ch in "SKDG":
                ax.text(x, y, ch, ha="center", va="center",
                        fontsize=12, weight="bold")
    coords = [(state[0], state[1]) for state in plan["states"]]
    ax.plot([x for x, _ in coords], [y for _, y in coords],
            color="#277da1", linewidth=2, alpha=0.8)
    ax.scatter([x for x, _ in coords], [y for _, y in coords],
               s=24, color="#277da1", zorder=4)
    ax.set_xlim(-0.5, len(GRID[0]) - 0.5)
    ax.set_ylim(len(GRID) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(GRID[0])))
    ax.set_yticks(range(len(GRID)))
    ax.set_title("先拿 K，才能经过 D")
    ax.grid(alpha=0.1)
    names = ["BFS", "A*: goal distance", "A*: through key"]
    counts = [methods[name]["expanded"] for name in names]
    axes[1].bar(range(len(names)), counts, color=["#517f93", "#8aa6b5", "#a54d13"])
    axes[1].set_xticks(range(len(names)), ["广度优先", "A*：只看目标", "A*：先拿钥匙"])
    axes[1].set_ylabel("扩展的完整状态数")
    axes[1].set_title("同一套规则，不同探索顺序")
    axes[1].tick_params(axis="x", labelrotation=8)
    axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle("路径只在地图模型里验过，未驱动真实机器",
                 fontsize=13)
    fig.tight_layout()
    target = WORK / "figures/key_door_search.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def format_states(states):
    return [
        {"x": x, "y": y, "has_key": key}
        for x, y, key in states
    ]


def main():
    states = all_states()
    heuristic_checks = verify_heuristics(states)
    bfs_result = bfs()
    bad_result = bfs(only_position=True)
    goal_astar = astar(goal_distance)
    key_astar = astar(key_aware_distance)
    constraint = constraint_plan()
    if bfs_result["actions"] is None:
        raise RuntimeError("有效 BFS 无法完成已写世界")
    shortest = len(bfs_result["actions"])
    for name, result in (
        ("A*: goal distance", goal_astar),
        ("A*: through key", key_astar),
        ("CP-SAT", constraint),
    ):
        if len(result["actions"]) != shortest:
            raise ValueError(f"{name} 的路线长度不等于 BFS 最短步数")
        replay(result["actions"])
    if constraint["first_feasible_horizon"] != shortest:
        raise ValueError("约束模型首次可行长度与 BFS 不同")
    if bad_result["actions"] is not None:
        raise ValueError("本图中仅按位置去重竟得到计划，需重审状态反例")
    methods = {
        "BFS": bfs_result,
        "A*: goal distance": goal_astar,
        "A*: through key": key_astar,
    }
    report = {
        "protocol": "work/verification/C11_key_door_protocol.md",
        "world_identity": "author-made 7x9 key-door grid; planning and replay only",
        "grid": GRID,
        "action_order": [name for name, _ in ACTIONS],
        "state_fields": ["x", "y", "has_key"],
        "start": format_states([START])[0],
        "goal": format_states([GOAL])[0],
        "shortest_steps": shortest,
        "methods": {
            name: {
                **{k: v for k, v in result.items() if k != "states"},
                "states": format_states(result["states"]),
            }
            for name, result in methods.items()
        },
        "position_only_visited_invalid": bad_result,
        "heuristic_checks": heuristic_checks,
        "constraint": {
            **{k: v for k, v in constraint.items() if k != "states"},
            "states": format_states(constraint["states"]),
        },
        "figure": "work/figures/key_door_search.png",
        "scope": "Finite author-made world. BFS/A* search and OR-Tools CP-SAT constraints use the same successor rules; replay validates plans in this model only, not physical execution or general AI ability.",
        "python": platform.python_version(),
        "ortools": ortools.__version__,
    }
    result_dir = WORK / "results"
    result_dir.mkdir(exist_ok=True)
    (result_dir / "key_door_planning.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    draw_figure(bfs_result, methods)
    print("最短步数:", shortest)
    print("BFS 行动:", "".join(bfs_result["actions"]))
    for name, result in methods.items():
        print(name, "扩展", result["expanded"],
              "发现", result["discovered"], "队列峰值", result["peak_frontier"])
    print("仅按位置去重:", bad_result["actions"], "扩展", bad_result["expanded"])
    print("CP-SAT 首次可行 T:", constraint["first_feasible_horizon"],
          "前面长度状态:", [x["status"] for x in constraint["statuses"][:-1]])


if __name__ == "__main__":
    main()
