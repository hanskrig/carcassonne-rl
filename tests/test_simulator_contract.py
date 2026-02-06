import random

from carc_rl.sim import CarcassonneSim


def test_contract_basic_step_and_actions():
    sim = CarcassonneSim(players=2)
    state = sim.reset(seed=123)
    assert state is not None

    actions = sim.legal_actions(state)
    assert isinstance(actions, list)
    assert len(actions) > 0

    next_state = sim.step(state, actions[0], random.Random(0))
    assert next_state is not None


def test_random_play_no_crash_with_move_cap():
    sim = CarcassonneSim(players=2)
    rng = random.Random(42)
    state = sim.reset(seed=42)

    max_moves = 300
    moves = 0
    while not sim.is_terminal(state) and moves < max_moves:
        actions = sim.legal_actions(state)
        assert actions
        state = sim.step(state, rng.choice(actions), rng)
        moves += 1

    assert moves > 0
