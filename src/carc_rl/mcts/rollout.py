"""Rollout policy for MCTS."""

from __future__ import annotations

import random
from typing import Any


def rollout_random(sim: Any, state: Any, root_player: int, rng: random.Random, max_depth: int = 256) -> float:
    rollout_state = sim.clone(state)
    steps = 0
    while not sim.is_terminal(rollout_state) and steps < max_depth:
        actions = sim.legal_actions(rollout_state)
        if not actions:
            break
        action = rng.choice(actions)
        rollout_state = sim.step(rollout_state, action, rng)
        steps += 1
    return sim.outcome(rollout_state, root_player)
