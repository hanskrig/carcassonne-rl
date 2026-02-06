"""Evaluation utilities for matches between agents."""

from __future__ import annotations

import random
import time
from typing import Any


def play_game(sim: Any, agentA: Any, agentB: Any, seed: int, max_moves: int = 1000) -> dict[str, Any]:
    rng = random.Random(seed)
    state = sim.reset(seed=seed)
    agents = [agentA, agentB]
    moves = 0

    while not sim.is_terminal(state) and moves < max_moves:
        player = sim.current_player(state)
        action = agents[player].select_action(sim, state, rng)
        state = sim.step(state, action, rng)
        moves += 1

    score_diff = sim.outcome(state, perspective_player=0)
    winner = 0 if score_diff > 0 else 1 if score_diff < 0 else -1
    return {
        "winner": winner,
        "score_diff": score_diff,
        "length": moves,
        "terminal": sim.is_terminal(state),
        "state": state,
    }


def run_match(sim: Any, agentA: Any, agentB: Any, n_games: int, seed: int = 0, max_moves: int = 1000) -> dict[str, Any]:
    start = time.perf_counter()
    wins = 0
    draws = 0
    total_diff = 0.0
    total_length = 0.0

    for idx in range(n_games):
        result = play_game(sim, agentA, agentB, seed=seed + idx, max_moves=max_moves)
        if result["winner"] == 0:
            wins += 1
        elif result["winner"] == -1:
            draws += 1
        total_diff += result["score_diff"]
        total_length += result["length"]

    runtime = time.perf_counter() - start
    return {
        "games": n_games,
        "win_rate_agentA": wins / n_games,
        "draw_rate": draws / n_games,
        "avg_score_diff_agentA": total_diff / n_games,
        "avg_length": total_length / n_games,
        "runtime_sec": runtime,
    }
