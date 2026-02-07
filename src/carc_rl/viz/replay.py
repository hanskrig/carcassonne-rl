"""Replay helpers for logged games."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def replay_game(sim: Any, game_record: dict[str, Any], max_steps: int | None = None, every: int = 1) -> None:
    """Replay a game log by matching recorded action keys to legal actions."""
    meta = game_record.get("meta", {})
    steps = game_record.get("steps", [])
    seed = meta.get("seed", 0)
    state = sim.reset(seed=seed)
    total_steps = len(steps) if max_steps is None else min(max_steps, len(steps))
    print(sim.render_text(state))
    for idx in range(total_steps):
        step = steps[idx]
        action_key = step.get("action_key")
        legal = sim.legal_actions(state)
        action = None
        for candidate in legal:
            if sim.action_to_key(candidate) == action_key:
                action = candidate
                break
        if action is None:
            print(f"[replay] Could not match action at step {idx}: {action_key}")
            break
        state = sim.step(state, action, rng=None)
        if (idx + 1) % every == 0:
            print(sim.render_text(state))


def plot_score_timeline(game_record: dict[str, Any]):
    steps = game_record.get("steps", [])
    diffs = []
    for step in steps:
        scores = step.get("scores_after", [])
        if isinstance(scores, dict):
            scores_list = list(scores.values())
        else:
            scores_list = scores
        if len(scores_list) >= 2:
            diffs.append(scores_list[0] - scores_list[1])
        elif scores_list:
            diffs.append(scores_list[0])
        else:
            diffs.append(0.0)
    fig, ax = plt.subplots()
    ax.plot(range(len(diffs)), diffs, marker="o", linewidth=1.5)
    ax.set_title("Per-Turn Score Difference (A - B)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Score Difference")
    fig.tight_layout()
    return fig
