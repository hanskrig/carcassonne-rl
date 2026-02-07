import random

from carc_rl.agents.random_agent import RandomAgent
from carc_rl.engine_adapter import action_key
from carc_rl.eval import play_game
from carc_rl.mcts.uct import MCTSAgent
from carc_rl.sim import CarcassonneSim


def test_mcts_can_select_initial_action():
    sim = CarcassonneSim(players=2)
    state = sim.reset(seed=1)
    mcts = MCTSAgent(n_simulations=1, seed=1)
    action = mcts.select_action(sim, state, random.Random(1))

    legal_keys = {action_key(a) for a in sim.legal_actions(state)}
    assert action_key(action) in legal_keys


def test_mcts_vs_random_smoke_match_runs():
    sim = CarcassonneSim(players=2)
    mcts = MCTSAgent(n_simulations=1, seed=2)
    rnd = RandomAgent()

    result, _ = play_game(sim, mcts, rnd, seed=2, max_moves=40)
    assert result["n_moves"] > 0
    assert "score_diff" in result
