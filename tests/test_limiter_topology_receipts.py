"""Receipt pins for wall-limited and diverted boundary modes."""

from __future__ import annotations

import json

import pytest

from benchmarks import limiter_topology_receipts as benchmark
from nova.equilibrium.topology import BoundaryMode, topology_solve_receipt


@pytest.fixture(scope="module")
def receipt():
    """Build the measured receipt once for the focused suite."""

    return benchmark.build_receipt()


def test_wall_contact_tolerance_is_fixed_before_scoring(receipt):
    tolerance = receipt["tolerances_fixed_before_scoring"][
        "wall_contact_distance_grid_cells"
    ]
    measured = receipt["synthetic_limited_round_trip"]
    assert tolerance == benchmark.WALL_CONTACT_TOLERANCE_GRID_CELLS == 1.0
    assert measured["tolerance_grid_cells"] == tolerance


def test_synthetic_limited_round_trip_recovers_contact_within_one_cell(receipt):
    measured = receipt["synthetic_limited_round_trip"]
    assert measured["passes"]
    assert measured["contact_distance_grid_cells"] <= 1.0
    limited = next(
        row
        for row in receipt["solves"]
        if row["solve_id"] == "synthetic-limited-round-trip"
    )
    assert limited["topology_class"] == BoundaryMode.LIMITED.value
    assert limited["boundary_point_m"] == limited["wall_contact_point_m"]


def test_every_solve_publishes_a_topology_class(receipt):
    solves = receipt["solves"]
    published = receipt["summary"]["topology_class_by_solve"]
    assert len(published) == receipt["summary"]["solve_count"] == len(solves)
    assert set(published.values()) <= {"limited", "diverted"}
    assert set(published) == {row["solve_id"] for row in solves}


def test_transition_history_counts_only_successful_traversal(receipt):
    transition = next(
        row
        for row in receipt["solves"]
        if row["solve_id"] == "synthetic-transition-traversal"
    )
    assert transition["topology_history"] == [
        "limited",
        "diverted",
        "limited",
        "diverted",
    ]
    assert transition["transition_count"] == 3
    assert transition["transitions_without_solver_failure"] == 3
    assert transition["solver_succeeded"]
    assert (
        receipt["summary"]["mid_solve_transitions_traversed_without_solver_failure"]
        == 3
    )


def test_failed_solve_does_not_bank_successful_transitions(receipt):
    states = benchmark._synthetic_limited_round_trip()[0], benchmark._diverted_state()
    failed = topology_solve_receipt(states, solver_succeeded=False)
    assert failed.transition_count == 1
    assert failed.transitions_without_solver_failure == 0
    assert not failed.solver_succeeded


def test_mast_limited_bank_agrees_without_efit_inputs(receipt):
    summary = receipt["summary"]
    assert summary["mast_limited_frame_count"] == len(benchmark.MAST_FRAME_PARAMETERS)
    assert summary["mast_limited_classifier_agreement_count"] == len(
        benchmark.MAST_FRAME_PARAMETERS
    )
    assert summary["mast_limited_classifier_agreement_fraction"] == 1.0
    assert receipt["mast_frame_provenance"]["efit_inputs_used"] is False
    json.dumps(receipt, allow_nan=False)
