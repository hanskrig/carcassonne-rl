"""MCTS node data structure."""

from __future__ import annotations

from typing import Any


class Node:
    __slots__ = (
        "state",
        "player_to_act",
        "parent",
        "parent_action_id",
        "children",
        "visits",
        "value_sum",
        "actions_list",
        "action_id_map",
        "full_action_ids",
        "untried_action_ids",
        "expanded_action_ids",
    )

    def __init__(
        self,
        state: Any,
        player_to_act: int,
        parent: "Node | None" = None,
        parent_action_id: int | None = None,
    ) -> None:
        self.state = state
        self.player_to_act = player_to_act
        self.parent = parent
        self.parent_action_id = parent_action_id
        self.children: dict[int, "Node"] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.actions_list: list[Any] | None = None
        self.action_id_map: dict[Any, int] | None = None
        self.full_action_ids: list[int] | None = None
        self.untried_action_ids: list[int] | None = None
        self.expanded_action_ids: set[int] = set()

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0
