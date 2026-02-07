"""Simulator contract for Carcassonne."""

from __future__ import annotations

import random
from typing import Any

from carc_rl import engine_adapter as adapter


class CarcassonneSim:
    """UI-free simulator wrapper around the engine."""

    def __init__(self, players: int = 2):
        self.players = players
        self.engine_name = "wingedsheep"
        self.ruleset = "base+farmers"

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

    def enumerate_actions(self, state: Any) -> tuple[list[Any], dict[tuple, int]]:
        """Return actions list and a stable action_id map."""
        actions = adapter.get_legal_actions(state)
        action_id_map: dict[tuple, int] = {}
        for idx, action in enumerate(actions):
            action_id_map[adapter.action_key(action)] = idx
        return actions, action_id_map

    def step(self, state: Any, action: Any, rng: random.Random | None = None) -> Any:
        del rng
        return adapter.apply_action(state, action)

    def is_terminal(self, state: Any) -> bool:
        return adapter.is_terminal(state)

    def outcome(self, state: Any, perspective_player: int) -> float:
        return adapter.outcome(state, perspective_player)

    def get_scores(self, state: Any) -> list[float] | dict[str, float]:
        scores = getattr(state, "scores", None)
        if scores is None:
            return []
        return list(scores)

    def action_to_key(self, action: Any) -> str:
        return repr(adapter.action_key(action))

    def action_to_repr(self, action: Any) -> str:
        return str(action)

    def action_extra(self, action: Any) -> dict[str, Any]:
        del action
        return {}

    def heuristic_value(self, state: Any, perspective_player: int) -> float:
        """Cheap heuristic: score diff minus a small frontier penalty."""
        score = self.outcome(state, perspective_player)
        board = getattr(state, "board", None)
        if board is None:
            return score
        rows = len(board)
        cols = len(board[0]) if rows else 0
        frontier = 0
        for r in range(rows):
            for c in range(cols):
                if board[r][c] is None:
                    continue
                if r > 0 and board[r - 1][c] is None:
                    frontier += 1
                if r < rows - 1 and board[r + 1][c] is None:
                    frontier += 1
                if c > 0 and board[r][c - 1] is None:
                    frontier += 1
                if c < cols - 1 and board[r][c + 1] is None:
                    frontier += 1
        return score - 0.01 * frontier

    def render_text(self, state: Any) -> str:
        remaining = len(getattr(state, "deck", []))
        return (
            f"player={state.current_player} phase={getattr(state.phase, 'name', state.phase)} "
            f"scores={state.scores} meeples={state.meeples} remaining_tiles={remaining}"
        )
