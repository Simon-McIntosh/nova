"""Tests for constrained-root LCFS attribution helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.efit_parity_root_geometry import (
    ORDER_OF_MAGNITUDE,
    _closed_axis_branch,
    _contour_geometry,
    _distance_pair,
)


def test_contour_geometry_exposes_sampling_closure_and_extent() -> None:
    points = np.asarray([[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, -1.0]])

    record = _contour_geometry(points)

    assert record["point_count"] == 4
    assert record["total_arclength_m"] == 3.0 + np.sqrt(5.0)
    assert record["endpoint_closure_gap_m"] == 0.0
    assert record["bounding_box_m"] == {
        "r_min": 1.0,
        "r_max": 2.0,
        "z_min": -1.0,
        "z_max": 1.0,
    }


def test_distance_pair_reproduces_symmetric_unordered_mean() -> None:
    left = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    right = np.asarray([[0.0, 1.0], [1.0, 1.0]])

    record = _distance_pair(left, right)

    assert record["symmetric_mean_distance_m"] == 1.0
    assert record["left_to_right"]["median_m"] == 1.0
    assert record["right_to_left"]["p90_m"] == 1.0
    assert record["correspondence_constraint"].startswith("none")


def test_closed_axis_branch_rejects_longer_open_component() -> None:
    closed = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )
    open_branch = np.asarray(
        [[-4.0, 2.0], [-2.0, 2.0], [0.0, 2.0], [2.0, 2.0], [4.0, 2.0]]
    )

    selected = _closed_axis_branch([open_branch, closed], np.asarray([0.0, 0.0]))

    assert np.array_equal(selected, closed)
    assert ORDER_OF_MAGNITUDE == 10.0


def test_banked_receipt_attributes_open_longest_component() -> None:
    receipt = json.loads(
        Path(
            "docs/figures/efit-forward-parity/converged-root-geometry-attribution.json"
        ).read_text()
    )
    record = receipt["lcfs_shape_match_attribution"]

    assert record["classification"] == "METRIC_SELECTION_ARTIFACT"
    assert record["aggregate_over_closed_branch_distance_ratio"] >= 10.0
    assert (
        record["aggregate_longest_branch"]["solved_contour"]["endpoint_closure_gap_m"]
        == 4.000000016157752
    )
    assert (
        record["closed_branch_enclosing_solved_axis"]["solved_contour"][
            "endpoint_closure_gap_m"
        ]
        == 0.0
    )
    assert receipt["banked_artifact_integrity"]["verified_digest_count"] == 23
    assert receipt["constrained_root"]["terminal_state"]["value_count"] == 1126
