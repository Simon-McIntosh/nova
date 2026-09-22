"""Receipt and boundary-authority rules for the topology parity harness."""

import numpy as np

from benchmarks.topology_parity_harness import (
    adjudicate_classification,
    adjudicate_difference,
    committed_class_authority,
    differing_cell_indices,
    differing_cell_records,
    marginal_status,
    panel_labels,
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
        "replay_completed": True,
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


def _coverage_receipt(rows):
    return {
        "schema": "nova.topology-cell-parity",
        "row_count": len(rows),
        "replayed_row_count": sum(
            row.get("replay_completed") is True for row in rows
        ),
        "unavailable_row_count": sum(
            row.get("replay_completed") is False for row in rows
        ),
        "not_replayable_row_count": sum(
            not row.get("replayable") for row in rows
        ),
        "rows": rows,
    }


def test_every_declared_row_missing_from_the_counts_is_a_coverage_defect():
    rows = [_replay_row(True), _replay_row(True)]
    receipt = _coverage_receipt(rows)
    assert receipt_errors(receipt) == []
    receipt["replayed_row_count"] = 1
    assert receipt_errors(receipt) == [
        "every declared row must be replayed or named unavailable"
    ]
    kept = rows[0]
    receipt["row_count"] = 1
    receipt["replayed_row_count"] = 1
    receipt["rows"] = [kept]
    assert receipt_errors(receipt) == []
    receipt["rows"] = [kept, rows[1]]
    assert receipt_errors(receipt) == ["row_count does not match the declared rows"]


def test_one_cell_label_difference_is_counted_and_adjudicated_never_absorbed():
    committed = np.array([0, 1, 1, 2, 3])
    replayed = np.array([0, 1, 0, 2, 3])
    coordinate = np.c_[np.linspace(0.0, 1.0, 5), np.zeros(5)]
    values = np.array([0.0, 0.5, 0.9, 1.1, 1.4])
    indices = differing_cell_indices(committed, replayed)
    assert indices.tolist() == [2]
    records = differing_cell_records(
        indices,
        coordinate,
        committed,
        replayed,
        values,
        adjudicate=lambda flux: adjudicate_difference(flux, 0.0, 1.0, 0.8),
    )
    assert len(records) == 1
    cell = records[0]
    assert cell["index"] == 2
    assert cell["centroid_m"] == coordinate[2].tolist()
    assert cell["committed_label"] == 1 and cell["replayed_label"] == 0
    assert cell["committed_label_name"] != cell["replayed_label_name"]
    assert cell["psi_norm_census_saddle"] == 0.9
    assert cell["psi_norm_raster_binding"] == 1.125
    assert cell["adjudication"] == "binding-level difference"


def test_committed_class_reads_the_operand_and_reports_receipt_conflicts():
    row = {"class": "diverted", "solve_topology_class": "diverted"}
    variants = {
        "receipt-a": {"achieved_class": "diverted"},
        "receipt-b": {"achieved_class": "limited"},
        "receipt-c": {"achieved_class": None},
    }
    committed, source, records, disagreement = committed_class_authority(
        row, {"achieved_class": "limited"}, variants, np.array([1.0, 2.0])
    )
    assert committed == "diverted"
    assert source == "operand bank metadata class"
    assert records["operand_metadata"] == "diverted"
    assert records["receipt-b"] == "limited" and records["receipt-c"] is None
    assert disagreement is True

    agreed = {"receipt-a": {"achieved_class": "diverted"}}
    committed, source, _, disagreement = committed_class_authority(
        row, {"achieved_class": "diverted"}, agreed, np.array([1.0, 2.0])
    )
    assert committed == "diverted" and disagreement is False

    _, source, _, _ = committed_class_authority(
        {}, {"achieved_class": "limited"}, {}, np.array([1.0, 2.0])
    )
    assert source == "governed receipt achieved_class"


def test_every_declared_row_has_panel_coverage_or_a_visible_annotation():
    replayed = ({"identity": "row one"}, {"coordinate": None})
    annotated = (
        {"identity": "row two", "replay_exception": "NoQualifiedAxisError"},
        {"annotation": "row two: NoQualifiedAxisError"},
    )
    declared = [replayed, annotated]
    labels = panel_labels(declared)
    assert len(labels) == len(declared)
    assert labels[0] == "row one"
    assert labels[1].startswith("row two") and "NoQualifiedAxisError" in labels[1]
