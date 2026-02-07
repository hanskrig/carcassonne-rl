"""Determinization UCT MCTS implementation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from carc_rl.engine_adapter import action_key
from carc_rl.mcts.node import Node
from carc_rl.mcts.rollout import rollout_random


class MCTSAgent:
    def __init__(
        self,
        n_simulations: int = 200,
        exploration_c: float = 1.4,
        rollout_depth: int = 16,
        seed: int | None = None,
    ):
        self.n_simulations = n_simulations
        self.exploration_c = exploration_c
        self.rollout_depth = rollout_depth
        self.seed = seed

    def _uct_score(self, parent_visits: int, child: Node) -> float:
        if child.visits == 0:
            return float("inf")
        return child.mean_value + self.exploration_c * math.sqrt(math.log(max(1, parent_visits)) / child.visits)

    def _select_child(self, node: Node) -> Node:
        return max(node.children.values(), key=lambda c: self._uct_score(node.visits, c))

    def select_action(self, sim: Any, state: Any, rng: random.Random | None = None) -> Any:
        rng = rng or random.Random(self.seed)
        root_player = sim.current_player(state)
        root_actions = sim.legal_actions(state)
        if not root_actions:
            raise RuntimeError("No legal actions at root")

        root_visit_counts: dict[tuple, int] = defaultdict(int)
        root_actions_by_key = {action_key(a): a for a in root_actions}

        for _ in range(self.n_simulations):
            det_state = sim.determinize_state(state, rng)
            root = Node(
                state=det_state,
                player_to_act=sim.current_player(det_state),
                untried_actions=sim.legal_actions(det_state),
            )

            node = root
            # Selection
            while not sim.is_terminal(node.state) and not node.untried_actions and node.children:
                node = self._select_child(node)

            # Expansion
            if not sim.is_terminal(node.state) and node.untried_actions:
                action = node.untried_actions.pop(rng.randrange(len(node.untried_actions)))
                next_state = sim.step(node.state, action, rng)
                child = Node(
                    state=next_state,
                    player_to_act=sim.current_player(next_state) if not sim.is_terminal(next_state) else -1,
                    parent=node,
                    parent_action=action,
                    untried_actions=sim.legal_actions(next_state) if not sim.is_terminal(next_state) else [],
                )
                node.children[action_key(action)] = child
                node = child

            value = rollout_random(sim, node.state, root_player, rng, max_depth=self.rollout_depth)

            # backprop
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

            # record first action visit for final move selection
            if root.children:
                top_child = next(iter(root.children.values()))
                root_visit_counts[action_key(top_child.parent_action)] += 1

        if not root_visit_counts:
            return rng.choice(root_actions)

        best_key = max(root_visit_counts, key=root_visit_counts.get)
        return root_actions_by_key[best_key]
