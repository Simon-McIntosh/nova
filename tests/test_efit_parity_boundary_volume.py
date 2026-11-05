import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.efit_parity_boundary_volume import (
    _polygon_measure,
    _require_bank_agreement,
    reconcile,
)
from benchmarks.efit_parity_root_geometry import (
    _boundary_branch_status,
    _closed_axis_branch,
)

SOURCE = Path(
    "docs/figures/efit-forward-parity/converged-root-geometry-attribution.json"
)
MOMENTS = Path("docs/figures/efit-forward-parity/moment-definition-rescore.json")
OUTPUT = Path(
    "docs/figures/efit-forward-parity/boundary-enclosed-volume-reconciliation.json"
)


def _source():
    return json.loads(SOURCE.read_text())


def _moments():
    return json.loads(MOMENTS.read_text())


def test_polygon_measure_reproduces_closed_form_revolution_volumes():
    """The quadrature is checked against analytic solids, not against itself."""
    outer, inner, bottom, top = 1.4, 0.6, -0.5, 0.7
    rectangle = np.array(
        [
            [outer, bottom],
            [outer, top],
            [inner, top],
            [inner, bottom],
        ]
    )
    measure = _polygon_measure(rectangle)
    analytic_area = (outer - inner) * (top - bottom)
    analytic_volume = np.pi * (outer**2 - inner**2) * (top - bottom)
    assert measure["poloidal_area_m2"] == pytest.approx(analytic_area, rel=1.0e-15)
    assert measure["area_centroid_major_radius_m"] == pytest.approx(
        0.5 * (outer + inner), rel=1.0e-15
    )
    assert measure["exact_solid_of_revolution_m3"] == pytest.approx(
        analytic_volume, rel=1.0e-15
    )

    centre, radius, count = 1.9, 0.35, 200_000
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    circle = np.column_stack(
        (centre + radius * np.cos(angle), radius * np.sin(angle))
    )
    measure = _polygon_measure(circle)
    assert measure["poloidal_area_m2"] == pytest.approx(
        np.pi * radius**2, rel=1.0e-9
    )
    assert measure["exact_solid_of_revolution_m3"] == pytest.approx(
        2.0 * np.pi**2 * radius**2 * centre, rel=1.0e-9
    )


def test_bank_agreement_guard_fires_only_on_a_closed_solved_boundary():
    """The guard is reachable: a closed boundary whose core volume is not the bank."""
    closed = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(RuntimeError, match="confined-core volume differs"):
        _require_bank_agreement(closed, 8.0, 9.0)
    assert _require_bank_agreement(closed, 8.0, 8.0) is True
    # With no closed boundary the comparison is undecidable, not refused.
    assert _require_bank_agreement(None, 8.0, 9.0) is False


@pytest.fixture(scope="module")
def receipt():
    return reconcile(SOURCE, MOMENTS)[0]


def test_open_unit_contour_is_refused_as_a_closed_boundary():
    branch = np.asarray([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    axis = np.asarray([0.25, 0.0])
    with pytest.raises(RuntimeError, match="no closed unit-boundary branch"):
        _closed_axis_branch([branch], axis)
    closed, display, status = _boundary_branch_status([branch], axis)
    assert closed is None
    np.testing.assert_array_equal(display, branch)
    assert status["status"] == "open"
    assert status["display_endpoint_gap_m"] == pytest.approx(2.0)


def test_stored_contour_keeps_its_revolution_quadrature(receipt):
    solved, stored = receipt["contour_comparison_table"]
    assert solved["status"] == "open"
    assert solved["poloidal_area_m2"] is None
    assert solved["exact_solid_of_revolution_m3"] is None
    assert receipt["boundary_status"]["status"] == "open"
    assert receipt["boundary_status"]["display_endpoint_gap_m"] > 0.0
    assert "separated endpoints" in receipt["boundary_status"]["reason"]
    assert stored["poloidal_area_m2"] > 0.0
    assert stored["area_centroid_major_radius_m"] > 0.0
    assert stored["exact_solid_of_revolution_m3"] == pytest.approx(
        stored["first_moment_approximation_m3"], rel=2.0e-14
    )


def test_banked_and_live_core_volumes_are_not_conflated(receipt):
    comparison = receipt["published_volume_comparison"]
    clipping = receipt["clipping"]
    banked_volume = _source()["moment_normalisation_attribution"][
        "solved_live_equilibrium"
    ]["plasma_volume_m3"]
    assert comparison["solved_m3"] == pytest.approx(banked_volume, rel=0.0, abs=0.0)
    assert clipping["banked_confined_core_toroidal_volume_m3"] == pytest.approx(
        banked_volume, rel=0.0, abs=0.0
    )
    assert clipping["reconstructed_core_matches_banked"] is False
    assert clipping["confined_core_toroidal_volume_m3"] != pytest.approx(
        comparison["solved_m3"], rel=2.0e-14
    )
    assert comparison[
        "reference_published_over_own_boundary_enclosed"
    ] == pytest.approx(1.0, rel=0.03)
    assert comparison["solved_published_over_own_boundary_enclosed"] is None
    assert comparison["disagreement_factor_boundary_over_published"] is None


def test_candidate_discrimination_requires_a_closed_solved_boundary(receipt):
    assert receipt["verdict"] == "UNDEFINED_OPEN_BOUNDARY"
    for candidate in receipt["candidate_discrimination"].values():
        assert candidate["verdict"] == "UNDEFINED_OPEN_BOUNDARY"
        assert candidate["reason"] == receipt["boundary_status"]["reason"]
    assert all(value is None for value in receipt["controlled_ratios"].values())


def test_core_current_remains_defined_but_area_fraction_does_not(receipt):
    clipping = receipt["clipping"]
    supports = _moments()["current_supports"]
    assert clipping["confined_core_cell_count"] < clipping["cell_count"]
    assert clipping["banked_confined_core_current_a"] == pytest.approx(
        supports["confined_core"], rel=0.0, abs=0.0
    )
    assert clipping["confined_core_current_a"] == clipping["live_core_current_a"]
    assert clipping["pinned_all_domain_current_a"] == pytest.approx(
        supports["all_domain_constrained"], rel=0.0, abs=0.0
    )
    assert clipping["banked_confined_core_current_fraction"] == pytest.approx(
        supports["confined_core_over_all_domain"], rel=0.0, abs=0.0
    )
    # The live fraction is checked against the independently banked currents
    # rather than against the benchmark's own division. The absolute bound is
    # the same 2e-9 A the benchmark pins its reconstructed currents to.
    # The live core carries nearly the whole reconstructed current, unlike the
    # banked 95-cell support; the fraction is checked against the live current
    # through the independently banked all-domain total.
    assert clipping["confined_core_current_a"] != pytest.approx(
        supports["confined_core"], abs=1.0
    )
    assert (
        clipping["confined_core_current_fraction"]
        * supports["all_domain_constrained"]
    ) == pytest.approx(clipping["confined_core_current_a"], rel=0.0, abs=2.0e-9)
    assert clipping["confined_core_area_fraction_of_solved_boundary"] is None
    assert clipping["area_fraction_minus_current_fraction"] is None


def test_boundary_support_rescore_is_undefined_for_open_contour(receipt):
    rescore = receipt["boundary_support_moment_rescore"]
    assert rescore["status"] == "undefined_open_boundary"
    assert rescore["metrics"]["poloidal_beta"][
        "original_signed_relative_deviation"
    ] == pytest.approx(-0.5628443566810165)
    assert rescore["metrics"]["internal_inductance"][
        "original_signed_relative_deviation"
    ] == pytest.approx(-0.7343445074393564)
    for metric in rescore["metrics"].values():
        assert metric["boundary_adjusted_solved_integral"] is None
        assert metric["signed_relative_deviation"] is None
        assert metric["deficit_eliminated"] is None
    assert set(rescore["parity_consequence"].values()) == {"UNDEFINED"}


def test_receipt_preserves_protected_evidence_and_identifies_historical_claim(receipt):
    assert receipt["execution_contract"]["nonlinear_solve_calls"] == 0
    assert receipt["protected_banked_artifacts"] == {
        "declared_count": 23,
        "verified_digest_count": 23,
        "all_digests_match": True,
        "source_and_output_receipts_are_outside_protected_set": True,
    }
    banked = json.loads(OUTPUT.read_text())
    assert (
        banked["verdict"]
        == "SOLVED_PUBLISHED_VOLUME_IS_CONFINED_CORE_NOT_BOUNDARY_ENCLOSED"
    )
    assert receipt["verdict"] == "UNDEFINED_OPEN_BOUNDARY"
