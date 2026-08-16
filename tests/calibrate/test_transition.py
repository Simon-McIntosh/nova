"""Simultaneous channels resolve common response but not local scalar causes."""

import numpy as np
import pytest

from nova.calibrate.transition import (
    TransitionError,
    apparent_block_count,
    evaluate_transition_discrimination,
    normalise_common_response,
)


CHANNELS = np.asarray(["target", "reference_a", "reference_b", "reference_c"])
BASELINES = np.asarray([1.0, 0.8, 1.3, 1.1])


def simultaneous_series(
    *, local_step: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pulse = np.arange(100, 110)
    shots = np.repeat(pulse, CHANNELS.size)
    channels = np.tile(CHANNELS, pulse.size)
    common = np.repeat(np.where(pulse % 2 == 0, 0.5, 2.0), CHANNELS.size)
    gains = np.tile(BASELINES, pulse.size) * common
    if local_step:
        gains[(channels == "target") & (shots >= 105)] *= 2.0
    return shots, channels, gains, np.full(gains.shape, 0.99)


def expected_target_step() -> list[dict]:
    return [
        {
            "after_scale": 2.0,
            "after_shot": 109,
            "before_scale": 1.0,
            "before_shot": 100,
            "channel": "target",
        }
    ]


def test_common_response_cancels_while_local_step_remains_exact():
    shots, channels, gains, shape = simultaneous_series()

    result = normalise_common_response(shots, channels, gains, shape_agreement=shape)
    target = result.channels == "target"

    assert result.corrected_gains[target & (result.shots < 105)] == pytest.approx(1.0)
    assert result.corrected_gains[target & (result.shots >= 105)] == pytest.approx(2.0)
    assert np.ptp(result.corrected_gains[result.channels == "reference_a"]) < 1.0e-12


def test_discrimination_recovers_injected_adjacent_transition():
    shots, channels, gains, shape = simultaneous_series()

    result = evaluate_transition_discrimination(
        shots,
        channels,
        gains,
        expected_target_step(),
        shape_agreement=shape,
    )

    assert result["refinement"]["exact_count"] == 1
    assert result["cause_counts"] == {"adjacent_transition": 1}
    assert result["corrected_apparent_blocks"] == len(CHANNELS) + 1


def test_common_response_step_disappears_from_every_channel():
    shots, channels, gains, shape = simultaneous_series(local_step=False)
    pulse_factor = np.where(shots < 105, 0.5, 2.0)
    gains = np.tile(BASELINES, 10) * pulse_factor

    result = normalise_common_response(shots, channels, gains, shape_agreement=shape)

    assert apparent_block_count(shots, channels, gains) == 8
    assert (
        apparent_block_count(result.shots, result.channels, result.corrected_gains) == 4
    )


def test_two_channels_cannot_define_a_robust_reference():
    with pytest.raises(TransitionError, match="at least three"):
        normalise_common_response([1, 1], ["a", "b"], [1.0, 2.0], minimum_peers=2)


def test_pair_multiplier_and_acquisition_multiplier_are_scalar_equivalent():
    shots, channels, gains, shape = simultaneous_series(local_step=False)
    acquisition = gains.copy()
    acquisition[(channels == "target") & (shots >= 105)] *= 2.0
    pair_state = gains.copy()
    pair_state[(channels == "target") & (shots >= 105)] *= 2.0

    acquisition_result = normalise_common_response(
        shots, channels, acquisition, shape_agreement=shape
    )
    pair_result = normalise_common_response(
        shots, channels, pair_state, shape_agreement=shape
    )

    assert acquisition_result.corrected_gains == pytest.approx(
        pair_result.corrected_gains
    )


def test_missing_state_labels_remain_visible_in_falsification_report():
    shots, channels, gains, shape = simultaneous_series()
    response_states = np.full(gains.shape, "")
    response_states[(channels == "reference_a") & (shots < 103)] = "both_members"

    result = evaluate_transition_discrimination(
        shots,
        channels,
        gains,
        expected_target_step(),
        shape_agreement=shape,
        response_states=response_states,
    )

    assert result["response_state_labels"] == 3
    assert result["configuration_state_labels"] == 0
