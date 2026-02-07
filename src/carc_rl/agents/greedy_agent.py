"""Greedy baseline agent."""

from __future__ import annotations

import random
from typing import Any


class GreedyAgent:
    """Pick action with highest immediate score delta for current player."""

    def select_action(self, sim: Any, state: Any, rng: random.Random) -> Any:
        player = sim.current_player(state)
        actions = sim.legal_actions(state)
        if not actions:
            raise RuntimeError("No legal actions available")

        best_action = None
        best_value = float("-inf")
        for action in actions:
            next_state = sim.step(sim.clone(state), action, rng)
            delta = next_state.scores[player] - state.scores[player]
            proxy = 1e-6 * rng.random()  # tie-breaker only
            value = float(delta) + proxy
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
