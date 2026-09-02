"""Receipt and boundary-authority rules for the topology parity harness."""

from benchmarks.topology_parity_harness import (
    adjudicate_difference,
    receipt_errors,
)


def test_boundary_authority_adjudication_on_manufactured_cell():
    """A cell between the two levels is a binding-level difference."""
    row = adjudicate_difference(
        flux=0.85,
        axis_flux=0.0,
        census_boundary_flux=1.0,
        raster_boundary_flux=0.8,
    )

    assert row["psi_norm_census_saddle"] == 0.85
    assert row["psi_norm_raster_binding"] == 1.0625
    assert row["census_closed"]
    assert not row["raster_closed"]
    assert row["adjudication"] == "binding-level difference"


def test_receipt_schema_requires_exact_non_marginal_parity():
    """Marginal differences need adjudication; ordinary rows need exact labels."""
    replayed = {
        "identity": "manufactured marginal",
        "replayable": True,
        "marginal_solver_basin": True,
        "compared_cell_count": 1,
        "differing_cell_count": 1,
        "differing_cells": [{"index": 0, "adjudication": "binding-level difference"}],
        "selected_primaries": {"axis": {"matches": False}},
        "classification": {"matches": False},
        "wall_node_census": {},
    }
    missing = {
        "identity": "manufactured unavailable",
        "replayable": False,
        "marginal_solver_basin": False,
        "not_replayable_reason": "no cached per-cell flux",
    }
    receipt = {
        "schema": "nova.topology-cell-parity",
        "rows": [replayed, missing],
    }

    assert receipt_errors(receipt) == []

    replayed["marginal_solver_basin"] = False
    assert receipt_errors(receipt) == [
        "manufactured marginal: non-marginal labels differ",
        "manufactured marginal: non-marginal primary differs",
        "manufactured marginal: non-marginal classification differs",
    ]
