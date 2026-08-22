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


def test_banked_receipt_applies_constrained_map_once_at_reference() -> None:
    receipt = json.loads(
        Path(
            "docs/figures/efit-forward-parity/converged-root-geometry-attribution.json"
        ).read_text()
    )
    record = receipt["constrained_map_at_reference_flux"]
    execution = record["execution_contract"]
    comparators = record["banked_comparators"]

    assert execution == {
        "nonlinear_solve_calls": 0,
        "constrained_map_primal_applications": 1,
        "tangent_applications": 0,
        "method": (
            "jax.linearize primal at the stored reference state, matching "
            "_composition_case_receipt with target_current supplied"
        ),
    }
    assert record["constraint"] == {
        "target_current_a": 933034.875,
        "prescribed_circuit_count": 101,
    }
    assert comparators["banked_mast_active_only_update"] == {
        "sup_fraction_of_span": 0.04075378153386053,
        "rms_fraction_of_span": 0.016584418612654646,
    }
    assert comparators["banked_wall_boundary_imbalance"] == {
        "before_passive_repair_fraction_of_span": 0.03440945100896484,
        "after_passive_repair_fraction_of_span": 0.004067584380328243,
    }
    assert record["decomposition_closure"]["sup_wb"] <= 5.0e-15
    assert receipt["banked_artifact_integrity"]["verified_digest_count"] == 23


def test_banked_receipt_uses_live_moment_normalisers() -> None:
    receipt = json.loads(
        Path(
            "docs/figures/efit-forward-parity/converged-root-geometry-attribution.json"
        ).read_text()
    )
    record = receipt["moment_normalisation_attribution"]
    solved = record["solved_live_equilibrium"]
    reference = record["reference"]

    assert record["execution_contract"] == {
        "nonlinear_solve_calls": 1,
        "terminal_state_serialisations_for_moments": 0,
        "equilibrium_reconstructions_from_serialised_state": 0,
        "moment_source": (
            "branch.equilibrium.moments read from the live equilibrium in "
            "the same process as the constrained row solve"
        ),
    }
    assert solved["constrained_terminal_cell_current_sum_a"] == 933034.875
    assert solved["moment_confined_core_current_a"] == 416958.2541259555
    assert np.isclose(
        solved["poloidal_beta_numerator_t2_m3"]
        / solved["common_boundary_normaliser_t2_m3"],
        solved["poloidal_beta"],
        rtol=2e-15,
    )
    assert np.isclose(
        solved["poloidal_field_squared_volume_integral_t2_m3"]
        / solved["common_boundary_normaliser_t2_m3"],
        solved["internal_inductance"],
        rtol=2e-15,
    )
    assert reference["normaliser_ratio_li_over_beta"] == 1.6318623846083615
    assert (
        record["common_or_independent"]
        == "COMMON_SOLVED_NORMALISER_DISTINCT_REFERENCE_NORMALISERS"
    )
    assert record["profile_amplitude_excluded"]["recovered_amplitude"] == (
        1.008098186771406
    )
