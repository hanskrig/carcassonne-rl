"""Random baseline agent."""

from __future__ import annotations

import random
from typing import Any


class RandomAgent:
    def select_action(self, sim: Any, state: Any, rng: random.Random) -> Any:
        actions = sim.legal_actions(state)
        if not actions:
            raise RuntimeError("No legal actions available")
        return rng.choice(actions)
