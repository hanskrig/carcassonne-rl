"""Evaluation utilities for matches between agents."""

from __future__ import annotations

import random
import time
from typing import Any

from carc_rl.logging.io import append_game_jsonl, write_summary_csv
from carc_rl.logging.recorder import GameRecorder


def play_game(
    sim: Any,
    agentA: Any,
    agentB: Any,
    seed: int,
    max_moves: int = 1000,
    recorder: GameRecorder | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    rng = random.Random(seed)
    state = sim.reset(seed=seed)
    agents = [agentA, agentB]
    moves = 0
    start = time.perf_counter()

    if recorder is not None:
        recorder.start_game(
            {
                "engine_name": getattr(sim, "engine_name", "unknown"),
                "ruleset": getattr(sim, "ruleset", "unknown"),
                "seed": seed,
                "agents": {
                    "A": getattr(agentA, "name", agentA.__class__.__name__),
                    "B": getattr(agentB, "name", agentB.__class__.__name__),
                },
            }
        )

    while not sim.is_terminal(state) and moves < max_moves:
        player = sim.current_player(state)
        scores_before = sim.get_scores(state)
        action = agents[player].select_action(sim, state, rng)
        state = sim.step(state, action, rng)
        scores_after = sim.get_scores(state)
        reward = 0.0
        if isinstance(scores_before, list) and isinstance(scores_after, list):
            if len(scores_after) > player and len(scores_before) > player:
                reward = scores_after[player] - scores_before[player]
        if recorder is not None:
            recorder.record_step(
                t=moves,
                player=player,
                action_key=sim.action_to_key(action),
                action_repr=sim.action_to_repr(action),
                reward=float(reward),
                scores_after=scores_after,
                extra=sim.action_extra(action),
            )
        moves += 1

    duration_ms = int((time.perf_counter() - start) * 1000)
    final_scores = sim.get_scores(state)
    score_diff = sim.outcome(state, perspective_player=0)
    winner = 0 if score_diff > 0 else 1 if score_diff < 0 else -1
    result = {
        "final_scores": final_scores,
        "winner": winner,
        "score_diff": score_diff,
        "n_moves": moves,
        "duration_ms": duration_ms,
        "terminal": sim.is_terminal(state),
    }
    record_dict = None
    if recorder is not None:
        recorder.end_game(result)
        record_dict = recorder.to_dict()
    return result, record_dict


def run_match(
    sim: Any,
    agentA: Any,
    agentB: Any,
    n_games: int,
    seed: int = 0,
    max_moves: int = 1000,
    record_games: bool = False,
    record_path_jsonl: str | None = None,
    record_path_csv: str | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    wins = 0
    draws = 0
    total_diff = 0.0
    total_length = 0.0
    records: list[dict[str, Any]] = []

    for idx in range(n_games):
        recorder = GameRecorder() if record_games else None
        result, record_dict = play_game(
            sim,
            agentA,
            agentB,
            seed=seed + idx,
            max_moves=max_moves,
            recorder=recorder,
        )
        if result["winner"] == 0:
            wins += 1
        elif result["winner"] == -1:
            draws += 1
        total_diff += result["score_diff"]
        total_length += result["n_moves"]
        if record_dict is not None:
            records.append(record_dict)
            if record_path_jsonl is not None:
                append_game_jsonl(record_path_jsonl, record_dict)

    runtime = time.perf_counter() - start
    if record_path_csv is not None and records:
        write_summary_csv(record_path_csv, records)
    return {
        "games": n_games,
        "win_rate_agentA": wins / n_games,
        "draw_rate": draws / n_games,
        "avg_score_diff_agentA": total_diff / n_games,
        "avg_length": total_length / n_games,
        "runtime_sec": runtime,
    }
