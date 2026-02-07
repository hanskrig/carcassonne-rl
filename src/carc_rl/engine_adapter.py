"""Adapter around the WingedSheep Carcassonne engine.

All engine-specific imports and logic should stay in this module.
"""

from __future__ import annotations

import copy
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _import_engine_symbols() -> dict[str, Any]:
    """Import engine symbols with a local-path fallback.

    Returns a mapping of imported names so downstream modules can avoid importing
    engine modules directly.
    """
    try:
        from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
        from wingedsheep.carcassonne.objects.actions.action import Action
        from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
        from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
        from wingedsheep.carcassonne.utils.action_util import ActionUtil
        from wingedsheep.carcassonne.utils.state_updater import StateUpdater
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        engine_path = repo_root / "engine"
        if str(engine_path) not in sys.path:
            sys.path.insert(0, str(engine_path))
        from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
        from wingedsheep.carcassonne.objects.actions.action import Action
        from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
        from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
        from wingedsheep.carcassonne.utils.action_util import ActionUtil
        from wingedsheep.carcassonne.utils.state_updater import StateUpdater

    return {
        "CarcassonneGameState": CarcassonneGameState,
        "Action": Action,
        "TileSet": TileSet,
        "SupplementaryRule": SupplementaryRule,
        "ActionUtil": ActionUtil,
        "StateUpdater": StateUpdater,
    }


ENGINE = _import_engine_symbols()
CarcassonneGameState = ENGINE["CarcassonneGameState"]
TileSet = ENGINE["TileSet"]
SupplementaryRule = ENGINE["SupplementaryRule"]
ActionUtil = ENGINE["ActionUtil"]
StateUpdater = ENGINE["StateUpdater"]


@contextmanager
def _temporary_random_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def new_game_state(seed: int, players: int = 2) -> Any:
    """Create a deterministic game state by controlling global random seed.

    The engine currently shuffles tiles through the module-level `random` API.
    """
    with _temporary_random_seed(seed):
        return CarcassonneGameState(
            players=players,
            tile_sets=(TileSet.BASE,),
            supplementary_rules=(SupplementaryRule.FARMERS,),
        )


def clone_state(state: Any) -> Any:
    """Deep copy an engine state for safe simulation branching."""
    return copy.deepcopy(state)


def get_current_player(state: Any) -> int:
    return int(state.current_player)


def get_legal_actions(state: Any) -> list[Any]:
    return list(ActionUtil.get_possible_actions(state))


def apply_action(state: Any, action: Any) -> Any:
    return StateUpdater.apply_action(state, action)


def is_terminal(state: Any) -> bool:
    return bool(state.is_terminated())


def outcome(state: Any, perspective_player: int) -> float:
    """Return score difference against the best opponent."""
    own = state.scores[perspective_player]
    opp = max(score for idx, score in enumerate(state.scores) if idx != perspective_player)
    return float(own - opp)


def action_key(action: Any) -> tuple:
    """Stable key for actions (engine objects use identity hashing)."""
    def _stable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return tuple(_stable(v) for v in value)
        if hasattr(value, "name") and hasattr(value, "value"):
            return (value.__class__.__name__, value.name)
        if hasattr(value, "row") and hasattr(value, "column"):
            return ("Coordinate", value.row, value.column)
        if hasattr(value, "description") and hasattr(value, "turns"):
            return ("Tile", value.description, value.turns)
        if hasattr(value, "__dict__"):
            return (
                value.__class__.__name__,
                tuple((k, _stable(v)) for k, v in sorted(vars(value).items())),
            )
        return str(value)

    return (
        action.__class__.__name__,
        tuple((name, _stable(getattr(action, name))) for name in sorted(vars(action).keys())),
    )
