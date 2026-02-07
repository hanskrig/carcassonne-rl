import random

from carc_rl.eval import play_game
from carc_rl.agents.random_agent import RandomAgent
from carc_rl.sim import CarcassonneSim


def test_clone_step_does_not_mutate_original():
    sim = CarcassonneSim(players=2)
    state = sim.reset(seed=7)
    state_clone = sim.clone(state)

    action = sim.legal_actions(state_clone)[0]
    _ = sim.step(state_clone, action, random.Random(0))

    assert state.scores == sim.clone(state).scores
    assert state.current_player == sim.clone(state).current_player


def test_seeded_random_vs_random_is_reproducible():
    sim = CarcassonneSim(players=2)
    a = RandomAgent()
    b = RandomAgent()

    r1, _ = play_game(sim, a, b, seed=99, max_moves=120)
    r2, _ = play_game(sim, a, b, seed=99, max_moves=120)

    assert r1["score_diff"] == r2["score_diff"]
    assert r1["n_moves"] == r2["n_moves"]
