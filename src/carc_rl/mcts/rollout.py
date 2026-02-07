"""Rollout policy for MCTS."""

from __future__ import annotations

import random
from typing import Any, Callable


def rollout_random(
    sim: Any,
    state: Any,
    root_player: int,
    rng: random.Random,
    max_depth: int = 60,
    heuristic_fn: Callable[[Any, int], float] | None = None,
) -> float:
    rollout_state = sim.clone(state)
    steps = 0
    while not sim.is_terminal(rollout_state) and steps < max_depth:
        actions = sim.legal_actions(rollout_state)
        if not actions:
            break
        action = rng.choice(actions)
        rollout_state = sim.step(rollout_state, action, rng)
        steps += 1
    if steps >= max_depth and heuristic_fn is not None:
        return heuristic_fn(rollout_state, root_player)
    return sim.outcome(rollout_state, root_player)
