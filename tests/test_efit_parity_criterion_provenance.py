from __future__ import annotations

import json

import pytest

from benchmarks import efit_parity_criterion_provenance as criterion
from nova.imas.parity_tolerances import (
    GEOMETRY_REFERENCE_IDENTITY,
    ScorecardField,
    registered_tolerances,
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return criterion.build_receipt()


def test_provenance_table_covers_only_carried_registered_bounds(receipt):
    rows = receipt["provenance_table"]
    tolerances = registered_tolerances()

    assert [row["field"] for row in rows] == [
        field.value for field in criterion.CARRIED_FIELDS
    ]
    assert len(rows) == 5
    for field, row in zip(criterion.CARRIED_FIELDS, rows, strict=True):
        tolerance = tolerances[field]
        assert row["bound"] == tolerance.bound
        assert row["direction"] == tolerance.direction.value
        assert row["unit"] == tolerance.unit
        assert row["basis_verbatim"] == tolerance.basis
        assert row["evidence_verbatim"] == tolerance.evidence


def test_provenance_classifies_physical_and_inherited_bounds(receipt):
    rows = {row["field"]: row for row in receipt["provenance_table"]}

    assert rows["magnetic_axis_distance_m"]["classification"] == "INHERITED"
    assert rows["lcfs_distance_m"]["classification"] == "INHERITED"
    assert rows["fixed_point_defect"]["classification"] == "INHERITED"
    assert rows["x_point_distance_m"]["classification"] == "PHYSICALLY-MOTIVATED"
    assert (
        rows["topology_class_agreement_fraction"]["classification"]
        == "PHYSICALLY-MOTIVATED"
    )
    assert rows["fixed_point_defect"]["evidence_read"]["relationship"] == (
        "merely-contains"
    )
    assert rows["x_point_distance_m"]["evidence_read"]["relationship"] == ("supports")


def test_registry_geometry_reference_identity_resolves():
    checked = json.loads(criterion.BOUND_CLASSIFICATION_OUTPUT.read_text())
    registered = registered_tolerances()
    geometry_fields = (
        ScorecardField.MAGNETIC_AXIS_DISTANCE_M,
        ScorecardField.LCFS_DISTANCE_M,
    )

    assert {registered[field].evidence for field in geometry_fields} == {
        GEOMETRY_REFERENCE_IDENTITY
    }
    assert checked["semantic_citation_contract"]["pinned_identity"] == (
        GEOMETRY_REFERENCE_IDENTITY
    )

    resolved = criterion.resolve_semantic_machine_artifact(
        checked["semantic_artifact_resolution"]["cache_directory"],
        GEOMETRY_REFERENCE_IDENTITY,
    )

    assert resolved["semantic_identity"] == GEOMETRY_REFERENCE_IDENTITY
    assert resolved["fully_verified"] is True


def test_units_audit_refuses_differently_normalised_comparison(receipt):
    audit = receipt["units_audit"]

    assert audit["fixed_point_residual"]["formula"] == (
        "max(abs(g(x) - x)) / max(max(abs(g(x))), 1e-30)"
    )
    assert (
        "grid, wall, and direct pre-clip sample-node"
        in audit["fixed_point_residual"]["denominator"]
    )
    assert audit["reference_side_scales"] == {
        "stored_lcfs_contour_discrepancy_fraction_of_declared_flux_span": 0.003456,
        "stored_lcfs_registered_limit_fraction_of_declared_flux_span": 0.006,
        "achieved_flux_agreement_rms_fraction_of_declared_flux_span": 0.00565,
        "span_definition": "peak-to-peak reference-grid total flux",
    }
    assert audit["comparability"]["verdict"] == "REFUSED"
    assert audit["comparability"]["numeric_conversion_supplied"] is False
    assert (
        "max(abs(g(x)))"
        in audit["comparability"]["exact_conversion_formula_if_inputs_existed"]
    )


def test_richardson_estimator_matches_banked_pair():
    estimate = criterion.richardson_fine_error(
        coarse_residual=0.010050357331518338,
        fine_residual=0.0015446543055689578,
        observed_order=2.7018908925836143,
    )

    assert estimate == pytest.approx(0.0015446543055689578, rel=1.0e-14)


def test_criterion_reports_five_estimates_and_one_stricter_existing_pass(receipt):
    rows = receipt["discretisation_consistent_criterion"]["per_reference"]
    estimates = [
        row["fine_mesh_richardson_error_estimate"]
        for row in rows
        if row["fine_mesh_richardson_error_estimate"] is not None
    ]

    assert len(rows) == 6
    assert len(estimates) == 5
    assert min(estimates) == pytest.approx(0.0013073838879026022)
    assert max(estimates) == pytest.approx(0.005299693489256675)
    assert all(row["passes_discretisation_consistent_criterion"] for row in rows)
    converged = next(row for row in rows if row["reference"] == "21986/46")
    assert converged["passes_registered_1e8_criterion"] is True
    assert converged["discretisation_consistent_criterion"] is None


def test_low_order_references_carry_per_reference_qualification(receipt):
    rows = {
        row["reference"]: row
        for row in receipt["discretisation_consistent_criterion"]["per_reference"]
    }

    for reference in ("21978/35", "22086/43"):
        assert rows[reference]["trust_qualification"].startswith("LEAST-TRUSTWORTHY")
        assert "outside any asymptotic regime" in rows[reference]["trust_qualification"]


def test_rescore_retains_both_counts_and_retracts_negative_figure(receipt):
    rescore = receipt["rescore"]

    assert rescore["registered_1e8_count_display"] == "1 of 6"
    assert rescore["discretisation_consistent_count_display"] == "6 of 6"
    assert rescore["both_counts_retained"] is True
    assert rescore["reliability_figure_survives"] is False


def test_claim_bounds_and_protected_digests_are_explicit(receipt):
    bounds = receipt["claim_bounds"]
    protected = receipt["protected_banked_artifacts"]

    assert bounds["new_equilibrium_solve"] is False
    assert bounds["equilibrium_solves_run"] == 0
    assert bounds["banked_residuals_only"] is True
    assert bounds["registered_tolerance_changed"] is False
    assert protected["verified_digest_count"] == 23
    assert protected["byte_for_byte_unchanged"] is True
    assert protected["before"]["mismatches"] == []
    assert protected["after"]["mismatches"] == []


def test_checked_receipt_matches_regeneration(tmp_path, receipt):
    checked = json.loads(criterion.OUTPUT_PATH.read_text())
    regenerated = criterion.write_receipt(tmp_path / "receipt.json")

    assert regenerated == receipt
    assert checked == regenerated
