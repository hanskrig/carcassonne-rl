"""Simulator contract for Carcassonne."""

from __future__ import annotations

import random
from typing import Any

from carc_rl import engine_adapter as adapter


class CarcassonneSim:
    """UI-free simulator wrapper around the engine."""

    def __init__(self, players: int = 2):
        self.players = players

    def reset(self, seed: int) -> Any:
        return adapter.new_game_state(seed=seed, players=self.players)

    def clone(self, state: Any) -> Any:
        return adapter.clone_state(state)

    def determinize_state(self, state: Any, rng: random.Random) -> Any:
        """Determinize hidden draw order by shuffling remaining deck copy.

        Engine mapping:
        - Preferred strategy implemented: sample remaining tile order in cloned state.
        - `next_tile` is already public/current and left unchanged.
        """
        cloned = self.clone(state)
        if hasattr(cloned, "deck") and isinstance(cloned.deck, list):
            rng.shuffle(cloned.deck)
        return cloned

    def current_player(self, state: Any) -> int:
        return adapter.get_current_player(state)

    def legal_actions(self, state: Any) -> list[Any]:
        return adapter.get_legal_actions(state)

    def step(self, state: Any, action: Any, rng: random.Random | None = None) -> Any:
        del rng
        return adapter.apply_action(state, action)

    def is_terminal(self, state: Any) -> bool:
        return adapter.is_terminal(state)

    def outcome(self, state: Any, perspective_player: int) -> float:
        return adapter.outcome(state, perspective_player)

    def render_text(self, state: Any) -> str:
        remaining = len(getattr(state, "deck", []))
        return (
            f"player={state.current_player} phase={getattr(state.phase, 'name', state.phase)} "
            f"scores={state.scores} meeples={state.meeples} remaining_tiles={remaining}"
        )
