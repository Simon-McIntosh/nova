"""Registered recovery gates for the closed-form equilibrium oracle."""

from __future__ import annotations

from scripts.oracle_rebaseline.gates import (
    EXPECTED_GATE_NAMES,
    load_report,
    validate_artifacts,
    validate_gauge_discipline,
    validate_registry,
)


def test_the_registry_covers_every_recovery_observable():
    """One registry carries every fixed-point and physics recovery check."""
    report = load_report()

    assert set(report["gate_registry"]) == EXPECTED_GATE_NAMES
    assert validate_registry(report)["passed"]
    for gate in report["gate_registry"].values():
        assert gate["status"] == "proposed"
        assert gate["measured_floor"] is not None
        assert gate["proposed_bound"] is not None
        assert gate["headroom"] is not None
        assert gate["convergence_clause"]["rejects_flat_above_floor"]


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


def test_serialized_terminal_roots_match_the_banked_receipt():
    """Root arrays, traces, and receipt digests agree with the merged result."""
    report = load_report()
    receipt = validate_artifacts(report)

    assert receipt["passed"]
    assert receipt["fixtures_checked"] == 2
    assert receipt["root_arrays_checked"] >= 8
