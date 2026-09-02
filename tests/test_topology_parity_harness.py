"""Receipt and boundary-authority rules for the topology parity harness."""

from benchmarks.topology_parity_harness import (
    adjudicate_classification,
    adjudicate_difference,
    marginal_status,
    parity_disposition,
    receipt_errors,
)


def test_boundary_authority_adjudication_on_manufactured_cell():
    row = adjudicate_difference(0.85, 0.0, 1.0, 0.8)
    assert row["psi_norm_census_saddle"] == 0.85
    assert row["psi_norm_raster_binding"] == 1.0625
    assert row["census_closed"] and not row["raster_closed"]
    assert row["adjudication"] == "binding-level difference"


def _replay_row(marginal):
    row = {
        "identity": "manufactured replay",
        "replayable": True,
        "marginal_solver_basin": marginal,
        "marginal_flag_source": "manufactured",
        "compared_cell_count": 1,
        "differing_cell_count": 1,
        "differing_cells": [{"index": 0, "adjudication": "binding-level difference"}],
        "selected_primaries": {"axis": {"matches": False}},
        "classification": {"finding": True},
        "wall_node_census": {},
    }
    row["disposition"] = parity_disposition(row)
    return row


def test_missing_marginal_flag_is_pending_and_never_an_exact_pass():
    marginal, source = marginal_status(None)
    assert marginal is None
    assert source == "missing solver_qualification.marginal_solver_basin"
    pending = _replay_row(marginal)
    receipt = {"schema": "nova.topology-cell-parity", "rows": [pending]}
    assert pending["disposition"] == "pending marginal qualification"
    assert receipt_errors(receipt) == []
    pending["marginal_solver_basin"] = False
    pending["disposition"] = parity_disposition(pending)
    assert receipt_errors(receipt) == [
        "manufactured replay: non-marginal labels differ",
        "manufactured replay: non-marginal primary differs",
        "manufactured replay: non-marginal classification differs",
    ]


def test_classification_disagreement_records_both_authorities_as_marginal():
    result = adjudicate_classification(
        committed="diverted",
        cell_authority="diverted",
        retained_raster="limited",
        cell_class_margin=float("inf"),
        retained_raster_class_margin=-0.125,
        cell_boundary_flux=-0.124,
        retained_raster_boundary_flux=-0.121,
        marginal=True,
    )
    assert result["matches_committed"] and not result["cell_raster_matches"]
    assert result["finding"] and result["cell_class_margin"] == "+Infinity"
    assert result["retained_raster_class_margin"] == -0.125
    assert result["cell_boundary_flux"] == -0.124
    assert result["retained_raster_boundary_flux"] == -0.121
    assert result["gate"] == "marginal finding"
    assert "production cell authority is diverted" in result["adjudication"]
    assert "retained raster is limited" in result["adjudication"]
