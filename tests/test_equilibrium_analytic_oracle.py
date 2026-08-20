"""Closed-form equilibrium fixtures with an exact prescribed exterior field."""

from __future__ import annotations

import numpy as np

from scripts.analytic_oracle_fixtures.measure import (
    ANALYTIC_CASE,
    analytic_case,
    boundary_read_receipt,
    cache_identity,
    import_audit,
)


def test_the_closed_form_boundary_is_the_production_boundary_read():
    """The topology locator reads the analytic zero-flux separatrix."""
    receipt = boundary_read_receipt()

    assert receipt["analytic_case"] == ANALYTIC_CASE
    assert receipt["topology_class"] == "limited"
    assert abs(receipt["closed_form_boundary_flux_wb"]) == 0.0
    assert (
        abs(
            receipt["production_boundary_flux_wb"]
            - receipt["closed_form_boundary_flux_wb"]
        )
        <= receipt["localisation_tolerance_wb"]
    )
    assert receipt["localisation_tolerance_wb"] < 1.0e-10
    assert receipt["axis_position_error_m"] < receipt["spatial_resolution_m"]


def test_the_fixture_identity_names_the_closed_form_oracle():
    """The semantic key cannot alias the stored-reference carrier."""
    case = analytic_case()
    identity = cache_identity(case, requested_cells=-500, wall_nodes=121)

    assert identity["schema"] == "analytic-oracle-hex-machine"
    assert identity["analytic_case"] == ANALYTIC_CASE
    assert identity["closed_form_constants"]["axis_flux_per_radian_wb"] == (
        case.axis_flux
    )
    assert identity["boundary_condition"] == "exact-analytic-exterior"
    assert "stored" not in repr(identity).lower()


def test_the_oracle_lane_has_no_stored_or_external_data_dependency():
    """The generating lane imports no map, archive, or IMAS reader."""
    audit = import_audit()

    assert audit["passed"]
    assert audit["forbidden_imports"] == []
    assert audit["forbidden_path_literals"] == []
    assert audit["closed_form_flux_self_check_relative"] < 8.0 * np.finfo(float).eps
