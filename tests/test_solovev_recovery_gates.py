"""Registered recovery gates for the closed-form equilibrium oracle."""

from __future__ import annotations

import pytest

from scripts.analytic_oracle_fixtures.measure import (
    FIXTURE_REQUESTS,
    TOTAL_FLUX_FACTOR,
    analytic_case,
    measure_fixture,
)
from scripts.analytic_oracle_fixtures.reduced_oracle import (
    convergence_clause_passes,
    measure_reduced_oracle,
)
from scripts.oracle_rebaseline.gates import (
    EXPECTED_GATE_NAMES,
    load_report,
    validate_artifacts,
    validate_gauge_discipline,
    validate_registry,
)


LOCKED_RECOVERY_BOUNDS = {
    "standing_forcing_sup_wb": 1.0e-13,
    "fixed_point_residual": 1.0e-12,
    "axis_position_m": 0.00525243590786495,
    "flux_sup_fraction_of_span": 1.0e-12,
    "flux_rms_fraction_of_span": 1.0e-12,
    "plasma_current_fraction": 0.011364810207658166,
    "poloidal_beta_fraction": 1.22739532336589,
    "internal_inductance_fraction": 1.2355549290537537,
    "field_integral_fraction": 0.016651149226035156,
    "grad_shafranov_relative": 0.0010130238719381077,
    "divergence_b_relative": 0.0023165108079651036,
    "divergence_j_relative": 0.0023446322707221623,
    "topology_class": "limited",
    "x_point_absence": "absent",
}
LOCKED_TANGENT_AXIS_RESPONSE_M = 1.0e-10
MAX_DEFAULT_LANE_SECONDS = 60.0


def test_the_reduced_oracle_is_a_seconds_class_roundoff_fixed_point():
    """The default lane cold-builds one real carrier and calls its map once."""
    receipt = measure_reduced_oracle()

    assert 100 <= receipt["realised_cells"] <= 150
    assert receipt["construction"] == (
        "closed-form field, profiles, density, and exact exterior"
    )
    assert receipt["map_evaluations"] == 1
    assert (
        receipt["forcing_sup_wb"] <= LOCKED_RECOVERY_BOUNDS["standing_forcing_sup_wb"]
    )
    assert (
        receipt["fixed_point_residual"]
        <= LOCKED_RECOVERY_BOUNDS["fixed_point_residual"]
    )
    assert receipt["wall_seconds"] < MAX_DEFAULT_LANE_SECONDS
    gauge = receipt["gauge_receipt"]
    assert gauge["raw_flux_comparison_gauge"] == "shared_exact_exterior"
    assert gauge["psi_norm_root_anchors_from"] == "root_field"
    assert gauge["psi_norm_oracle_anchors_from"] == "closed_form_field"
    assert not gauge["reference_gauge_constant_used"]


def test_a_flat_excess_fails_even_below_every_absolute_bound():
    """Convergence is mandatory even for an arbitrarily small excess."""
    assert not convergence_clause_passes(1.0e-16, 1.0e-16, 0.0, 0.0)
    assert convergence_clause_passes(1.0e-16, 1.0e-18, 0.0, 0.0)


@pytest.mark.slow
def test_the_registry_covers_every_recovery_observable():
    """One registry carries every fixed-point and physics recovery check."""
    report = load_report()

    assert set(report["gate_registry"]) == EXPECTED_GATE_NAMES
    assert set(report["gate_registry"]) == set(LOCKED_RECOVERY_BOUNDS)
    assert validate_registry(report)["passed"]
    for name, gate in report["gate_registry"].items():
        assert gate["status"] == "proposed"
        assert gate["proposed_bound"] == LOCKED_RECOVERY_BOUNDS[name]
        assert gate["measured_floor"] is not None
        assert gate["proposed_bound"] is not None
        assert gate["headroom"] is not None
        assert gate["convergence_clause"]["rejects_flat_above_floor"]


@pytest.mark.slow
def test_every_flux_comparison_obeys_one_gauge_discipline():
    """Flux amplitudes and normalised coordinates use their own field anchors."""
    report = load_report()
    receipt = validate_gauge_discipline(report)

    assert receipt["passed"]
    assert receipt["mixed_gauge_gates"] == []
    assert receipt["foreign_anchor_gates"] == []
    for fixture in report["fixtures"].values():
        assert fixture["seed"]["independent_of_closed_form_state"]
        assert fixture["gauge_receipt"]["psi_norm_root_anchors_from"] == "root_field"
        assert fixture["gauge_receipt"]["psi_norm_oracle_anchors_from"] == (
            "closed_form_field"
        )


@pytest.mark.slow
def test_independent_seed_roots_report_the_measured_recovery_verdict():
    """A small solver residual never hides an independently seeded wrong root."""
    report = load_report()
    registry = validate_registry(report)

    assert set(report["fixtures"]) == {"coarse", "fine"}
    for fixture in report["fixtures"].values():
        assert fixture["terminal_root"]["criterion_met"]
        assert fixture["terminal_root"]["topology_class"] == "limited"
        assert fixture["terminal_root"]["x_point"] is None
    if report["verdict"]["roundoff_class_recovery"]:
        assert report["verdict"]["all_proposed_gates_pass"]
        assert report["verdict"]["all_convergence_clauses_pass"]
        assert registry["all_bounds_pass"]
        assert registry["all_convergence_clauses_pass"]
    else:
        assert report["recovery_finding"]["status"] == "alternate-root-hold"
        assert not report["verdict"]["all_proposed_gates_pass"]
        assert not report["verdict"]["all_convergence_clauses_pass"]
        failed = set(registry["failed_bounds"]) | set(registry["failed_convergence"])
        assert {
            "axis_position_m",
            "flux_sup_fraction_of_span",
            "flux_rms_fraction_of_span",
        } <= failed


@pytest.mark.slow
def test_serialized_terminal_roots_match_the_banked_receipt():
    """Root arrays, traces, and receipt digests agree with the merged result."""
    report = load_report()
    receipt = validate_artifacts(report)

    assert receipt["passed"]
    assert receipt["fixtures_checked"] == 2
    assert receipt["root_arrays_checked"] >= 8


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", tuple(FIXTURE_REQUESTS))
def test_the_full_warm_fixture_keeps_the_roundoff_sentries(fixture_name):
    """The full carriers retain round-off forcing and picometre response pins."""
    receipt = measure_fixture(fixture_name, FIXTURE_REQUESTS[fixture_name])
    span = TOTAL_FLUX_FACTOR * analytic_case().axis_flux

    assert receipt["cache"]["hit"]
    assert (
        receipt["forcing"]["sup_wb"]
        <= LOCKED_RECOVERY_BOUNDS["standing_forcing_sup_wb"]
    )
    assert (
        receipt["forcing"]["sup_wb"] / span
        <= LOCKED_RECOVERY_BOUNDS["fixed_point_residual"]
    )
    assert (
        receipt["linear_response"]["projected_state"]["axis_displacement_mm"] * 1.0e-3
        <= LOCKED_TANGENT_AXIS_RESPONSE_M
    )
