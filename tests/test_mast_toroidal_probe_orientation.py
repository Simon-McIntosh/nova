"""Directed probe inference needs a held-out absolute-field agreement."""

import numpy as np

from benchmarks.mast_toroidal_probe_orientation import (
    infer_orientation,
    score_observations,
)


def test_flipped_probe_is_inferred_as_negative_phi() -> None:
    reference = np.array([0.2, 0.4, 0.6])
    result = infer_orientation(
        [(reference, -reference)],
        [(reference, -1.01 * reference)],
    )
    assert result["verdict"] == "promoted"
    assert result["orientation_sign"] == -1
    assert result["field_direction"] == "-e_phi"
    assert result["maximum_held_out_relative_error"] < 0.02


def test_excess_held_out_amplitude_error_stays_unresolved() -> None:
    reference = np.array([0.2, 0.4, 0.6])
    result = infer_orientation(
        [(reference, -reference)],
        [(reference, -1.03 * reference)],
    )
    assert result["verdict"] == "unresolved"
    assert result["orientation_sign"] is None
    assert result["candidate_sign"] == -1
    assert result["maximum_held_out_relative_error"] > 0.02


def test_missing_holdout_cannot_promote() -> None:
    reference = np.array([0.2, 0.4])
    result = infer_orientation([(reference, reference)], [])
    assert result["verdict"] == "unresolved"


def test_shot_partition_and_independent_sources() -> None:
    rows = [
        {
            "probe": "probe_a",
            "shot": shot,
            "reference_source": "sourced_tf_field",
            "measured_source": "probe_acquisition",
            "reference_T": [0.2, 0.4],
            "measured_T": [-0.2, -0.4],
        }
        for shot in range(10, 15)
    ]
    result = score_observations(rows, shots=list(range(10, 15)), probes=["probe_a"])[
        "probe_a"
    ]
    assert result["held_out_shot_ids"] == [14]
    assert result["training_shot_ids"] == [10, 11, 12, 13]
    assert result["orientation_sign"] == -1
