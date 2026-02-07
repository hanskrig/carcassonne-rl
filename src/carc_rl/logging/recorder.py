"""Game logging recorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from carc_rl.logging.schema import GameRecord, StepRecord


@dataclass
class GameRecorder:
    """Collect structured game logs for serialization."""

    meta: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    steps: list[StepRecord] = field(default_factory=list)

    def start_game(self, meta_dict: dict[str, Any]) -> None:
        meta = dict(meta_dict)
        meta.setdefault("game_id", str(uuid4()))
        meta.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        self.meta = meta
        self.steps = []
        self.result = None

    def record_step(
        self,
        t: int,
        player: int,
        action_key: str,
        action_repr: str,
        reward: float,
        scores_after: list[float] | dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.meta is None:
            raise RuntimeError("start_game must be called before record_step.")
        self.steps.append(
            StepRecord(
                t=t,
                player=player,
                action_key=action_key,
                action_repr=action_repr,
                reward=reward,
                scores_after=scores_after,
                extra=extra or {},
            )
        )

    def end_game(self, result_dict: dict[str, Any]) -> None:
        if self.meta is None:
            raise RuntimeError("start_game must be called before end_game.")
        self.result = dict(result_dict)

    def to_dict(self) -> dict[str, Any]:
        if self.meta is None or self.result is None:
            raise RuntimeError("GameRecorder is incomplete; missing meta or result.")
        record = GameRecord(meta=self.meta, result=self.result, steps=self.steps)
        return record.to_dict()
