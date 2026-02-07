"""Determinization UCT MCTS implementation."""

from __future__ import annotations

import math
import multiprocessing as mp
import random
from collections import defaultdict
from typing import Any, Iterable

from carc_rl.mcts.node import Node
from carc_rl.mcts.rollout import rollout_random


def _run_root_worker(payload: dict[str, Any]) -> dict[int, int]:
    sim = payload["sim"]
    state = payload["state"]
    seed = payload["seed"]
    n_simulations = payload["n_simulations"]
    batch_size = payload["batch_size"]
    params = payload["params"]
    agent = MCTSAgent(**params)
    rng = random.Random(seed)
    return agent._run_simulations(sim, state, rng, n_simulations, batch_size)


class MCTSAgent:
    def __init__(
        self,
        n_simulations: int = 200,
        exploration_c: float = 1.4,
        rollout_depth: int = 60,
        seed: int | None = None,
        batch_size: int = 100,
        progressive_widening: bool = True,
        widening_k_max: int = 50,
        widening_c: float = 2.5,
        widening_alpha: float = 0.5,
        root_parallel_workers: int = 0,
    ):
        self.n_simulations = n_simulations
        self.exploration_c = exploration_c
        self.rollout_depth = rollout_depth
        self.seed = seed
        self.batch_size = batch_size
        self.progressive_widening = progressive_widening
        self.widening_k_max = widening_k_max
        self.widening_c = widening_c
        self.widening_alpha = widening_alpha
        self.root_parallel_workers = root_parallel_workers

    def _uct_score(self, parent_visits: int, child: Node) -> float:
        if child.visits == 0:
            return float("inf")
        return child.mean_value + self.exploration_c * math.sqrt(math.log(max(1, parent_visits)) / child.visits)

    def _select_child(self, node: Node) -> Node:
        return max(node.children.values(), key=lambda c: self._uct_score(node.visits, c))

    def _ensure_actions(self, sim: Any, node: Node) -> None:
        if node.actions_list is not None:
            return
        actions, action_id_map = sim.enumerate_actions(node.state)
        node.actions_list = actions
        node.action_id_map = action_id_map
        node.full_action_ids = list(range(len(actions)))
        node.untried_action_ids = node.full_action_ids.copy()

    def _widening_limit(self, visits: int) -> int:
        if not self.progressive_widening:
            return self.widening_k_max
        return min(self.widening_k_max, max(1, int(self.widening_c * (visits ** self.widening_alpha))))

    def _expand_action_id(self, node: Node, rng: random.Random) -> int | None:
        if node.full_action_ids is None or node.untried_action_ids is None:
            return None
        limit = self._widening_limit(node.visits)
        if len(node.expanded_action_ids) >= limit:
            return None
        if not node.untried_action_ids:
            return None
        idx = rng.randrange(len(node.untried_action_ids))
        action_id = node.untried_action_ids.pop(idx)
        node.expanded_action_ids.add(action_id)
        return action_id

    def _run_simulations(
        self,
        sim: Any,
        state: Any,
        rng: random.Random,
        n_simulations: int,
        batch_size: int,
    ) -> dict[int, int]:
        root_visit_counts: dict[int, int] = defaultdict(int)
        root_player = sim.current_player(state)
        heuristic_fn = sim.heuristic_value

        sims_remaining = n_simulations
        while sims_remaining > 0:
            current_batch = min(batch_size, sims_remaining)
            sims_remaining -= current_batch
            det_state = sim.determinize_state(state, rng)

            for _ in range(current_batch):
                root_state = sim.clone(det_state)
                root = Node(
                    state=root_state,
                    player_to_act=sim.current_player(root_state),
                    parent=None,
                    parent_action_id=None,
                )
                node = root
                root_action_id = None

                # Selection
                while not sim.is_terminal(node.state):
                    self._ensure_actions(sim, node)
                    action_id = self._expand_action_id(node, rng)
                    if action_id is not None:
                        action = node.actions_list[action_id]
                        next_state = sim.step(node.state, action, rng)
                        child = Node(
                            state=next_state,
                            player_to_act=sim.current_player(next_state) if not sim.is_terminal(next_state) else -1,
                            parent=node,
                            parent_action_id=action_id,
                        )
                        node.children[action_id] = child
                        node = child
                        if root_action_id is None:
                            root_action_id = action_id
                        break
                    if not node.children:
                        break
                    node = self._select_child(node)
                    if node.parent is root and root_action_id is None:
                        root_action_id = node.parent_action_id

                # Rollout
                value = rollout_random(
                    sim,
                    node.state,
                    root_player,
                    rng,
                    max_depth=self.rollout_depth,
                    heuristic_fn=heuristic_fn,
                )

                # Backprop
                while node is not None:
                    node.visits += 1
                    node.value_sum += value
                    node = node.parent

                if root_action_id is not None:
                    root_visit_counts[root_action_id] += 1

        return root_visit_counts

    def _aggregate_counts(self, counts: Iterable[dict[int, int]]) -> dict[int, int]:
        merged: dict[int, int] = defaultdict(int)
        for count in counts:
            for key, value in count.items():
                merged[key] += value
        return merged

    def select_action(self, sim: Any, state: Any, rng: random.Random | None = None) -> Any:
        rng = rng or random.Random(self.seed)
        actions, action_id_map = sim.enumerate_actions(state)
        if not actions:
            raise RuntimeError("No legal actions at root")

        if self.root_parallel_workers and self.root_parallel_workers > 1:
            workers = self.root_parallel_workers
            per_worker = max(1, self.n_simulations // workers)
            payloads = []
            for idx in range(workers):
                payloads.append(
                    {
                        "sim": sim,
                        "state": state,
                        "seed": rng.randint(0, 10**9),
                        "n_simulations": per_worker,
                        "batch_size": self.batch_size,
                        "params": {
                            "n_simulations": per_worker,
                            "exploration_c": self.exploration_c,
                            "rollout_depth": self.rollout_depth,
                            "seed": rng.randint(0, 10**9),
                            "batch_size": self.batch_size,
                            "progressive_widening": self.progressive_widening,
                            "widening_k_max": self.widening_k_max,
                            "widening_c": self.widening_c,
                            "widening_alpha": self.widening_alpha,
                            "root_parallel_workers": 0,
                        },
                    }
                )
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=workers) as pool:
                    counts = pool.map(_run_root_worker, payloads)
            except Exception:
                counts = [self._run_simulations(sim, state, rng, self.n_simulations, self.batch_size)]
            root_visit_counts = self._aggregate_counts(counts)
        else:
            root_visit_counts = self._run_simulations(sim, state, rng, self.n_simulations, self.batch_size)

        if not root_visit_counts:
            return rng.choice(actions)

        best_id = max(root_visit_counts, key=root_visit_counts.get)
        return actions[best_id]
