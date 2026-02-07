"""Schema helpers for game logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """A single step within a game."""

    t: int
    player: int
    action_key: str
    action_repr: str
    reward: float
    scores_after: list[float] | dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameRecord:
    """Top-level game record container."""

    meta: dict[str, Any]
    result: dict[str, Any]
    steps: list[StepRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "result": self.result,
            "steps": [asdict(step) for step in self.steps],
        }
