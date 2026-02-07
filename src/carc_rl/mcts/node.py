"""MCTS node data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    state: Any
    player_to_act: int
    parent: "Node | None" = None
    parent_action: Any = None
    children: dict[tuple, "Node"] = field(default_factory=dict)
    untried_actions: list[Any] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0
