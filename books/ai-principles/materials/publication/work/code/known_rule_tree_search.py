"""C24: exact minimax and finite Monte Carlo tree search on known tiny rules.

This is a transparent search illustration, not AlphaGo, AlphaZero, or
evidence that a language model can solve a new task by self-play.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def outcome(board: tuple[int, ...]) -> int | None:
    for line in LINES:
        if board[line[0]] != 0 and board[line[0]] == board[line[1]] == board[line[2]]:
            return board[line[0]]
    return 0 if all(cell != 0 for cell in board) else None


def legal(board: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i for i, cell in enumerate(board) if cell == 0)


def after(board: tuple[int, ...], player: int, action: int) -> tuple[int, ...]:
    if outcome(board) is not None or board[action] != 0:
        raise ValueError("action illegal or game already over")
    next_board = list(board)
    next_board[action] = player
    return tuple(next_board)


@lru_cache(maxsize=None)
def exact_value(board: tuple[int, ...], player: int) -> int:
    end = outcome(board)
    if end is not None:
        return end * player
    return max(-exact_value(after(board, player, action), -player) for action in legal(board))


class Node:
    def __init__(self, board: tuple[int, ...], player: int):
        self.board = board
        self.player = player
        self.visits = 0
        self.value_sum = 0.0  # outcome from this node's player perspective
        self.children: dict[int, Node] = {}
        self.untried = list(legal(board))

    def best_child_for_search(self, exploration: float) -> tuple[int, "Node"]:
        if not self.children:
            raise RuntimeError("node has no explored child")
        return max(
            self.children.items(),
            key=lambda item: (
                -item[1].value_sum / item[1].visits
                + exploration * math.sqrt(math.log(self.visits) / item[1].visits)
            ),
        )


def monte_carlo_search(
    board: tuple[int, ...], player: int, simulations: int, seed: int
) -> dict:
    root = Node(board, player)
    rng = random.Random(seed)
    for _ in range(simulations):
        node = root
        path = [node]
        while outcome(node.board) is None and not node.untried:
            _, node = node.best_child_for_search(math.sqrt(2.0))
            path.append(node)
        if outcome(node.board) is None and node.untried:
            action = rng.choice(node.untried)
            node.untried.remove(action)
            child = Node(after(node.board, node.player, action), -node.player)
            node.children[action] = child
            node = child
            path.append(node)
        rollout_board, rollout_player = node.board, node.player
        while outcome(rollout_board) is None:
            action = rng.choice(legal(rollout_board))
            rollout_board = after(rollout_board, rollout_player, action)
            rollout_player = -rollout_player
        winner = outcome(rollout_board)
        assert winner is not None
        for visited in path:
            visited.visits += 1
            visited.value_sum += winner * visited.player

    counts = {action: child.visits for action, child in root.children.items()}
    root_estimates = {
        action: -child.value_sum / child.visits
        for action, child in root.children.items()
    }
    chosen = max(counts, key=lambda action: counts[action])
    return {
        "simulations": simulations,
        "seed": seed,
        "root_visits": root.visits,
        "move_visit_counts": counts,
        "root_player_estimated_values": root_estimates,
        "search_policy_from_visits": {
            action: count / simulations for action, count in counts.items()
        },
        "chosen_by_most_visits": chosen,
    }


def show_board(board: tuple[int, ...]) -> list[str]:
    char = {0: ".", 1: "X", -1: "O"}
    return [
        "".join(char[board[3 * row + col]] for col in range(3))
        for row in range(3)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("work/results/c24_known_rule_search.json")
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"preserve previous search result: {out}")
    # X to act. O threatens the top row; only blocking cell 2 avoids a loss.
    board = (-1, -1, 0, 0, 1, 0, 0, 1, 0)
    player = 1
    exact_actions = {
        action: -exact_value(after(board, player, action), -player)
        for action in legal(board)
    }
    searches = [
        monte_carlo_search(board, player, simulations, seed=20260924)
        for simulations in (20, 100, 500, 2000)
    ]
    for search in searches:
        action = search["chosen_by_most_visits"]
        search["exact_value_of_chosen_action"] = exact_actions[action]
        search["best_exact_value"] = max(exact_actions.values())
    result = {
        "scope": "complete public rules of 3x3 tic-tac-toe; exact minimax and random-rollout UCT; no learned policy/value network and no AlphaZero scale",
        "board": show_board(board),
        "board_internal": board,
        "player_to_act": "X",
        "terminal_payoff": "winner +1 for X, -1 for O; draw 0",
        "exact_action_values_from_X_perspective": exact_actions,
        "exact_best_value": max(exact_actions.values()),
        "searches": searches,
        "why_rules_matter": "each tree rollout executes hypothetical legal moves using a known transition and terminal-winner rule; unknown game dynamics would require a learned or provided simulator",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out)
    print("board", show_board(board))
    print("exact", exact_actions)
    for search in searches:
        print(search["simulations"], search["chosen_by_most_visits"], search["exact_value_of_chosen_action"], search["move_visit_counts"])


if __name__ == "__main__":
    main()
