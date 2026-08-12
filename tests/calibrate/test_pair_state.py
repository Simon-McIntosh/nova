"""Manufactured pickup-state sequences recover exact discrete states."""

import numpy as np

from nova.calibrate.pair_state import classify_pair_states


def test_stable_half_full_and_recovered_states_are_resolved():
    measured = np.repeat([0.5, 1.0, 1.5], 5)

    sequence = classify_pair_states(measured)

    assert [block.state for block in sequence.blocks] == [
        "single_member",
        "both_members",
        "recovered",
    ]
    assert [block.multiplier for block in sequence.blocks] == [0.5, 1.0, 1.5]
    assert sequence.transition_count == 2
    assert not sequence.unresolved


def test_one_persistent_change_is_a_midlife_step():
    measured = np.r_[np.full(8, 1.0), np.full(9, 0.5)]

    sequence = classify_pair_states(measured)

    assert sequence.midlife_step
    assert not sequence.flips
    assert [(block.start, block.stop) for block in sequence.blocks] == [(0, 8), (8, 17)]


def test_shot_to_shot_state_flips_are_preserved_not_averaged():
    measured = np.array([1.0, 0.5, 1.0, 0.5, 1.0])

    sequence = classify_pair_states(measured)

    assert sequence.flips
    assert sequence.transition_count == 4
    assert sequence.assignments == (
        "both_members",
        "single_member",
        "both_members",
        "single_member",
        "both_members",
    )


def test_values_between_declared_states_remain_unresolved():
    sequence = classify_pair_states([1.0, 0.78, 1.0])

    assert sequence.unresolved == (1,)
    assert sequence.assignments[1] is None
