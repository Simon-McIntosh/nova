import json
from pathlib import Path

import pytest

from benchmarks.efit_parity_boundary_volume import reconcile

SOURCE = Path(
    "docs/figures/efit-forward-parity/converged-root-geometry-attribution.json"
)
MOMENTS = Path("docs/figures/efit-forward-parity/moment-definition-rescore.json")
OUTPUT = Path(
    "docs/figures/efit-forward-parity/boundary-enclosed-volume-reconciliation.json"
)


@pytest.fixture(scope="module")
def receipt():
    return reconcile(SOURCE, MOMENTS)[0]


def test_both_contours_use_the_same_revolution_quadrature(receipt):
    rows = receipt["contour_comparison_table"]

    assert len(rows) == 2
    for row in rows:
        assert row["poloidal_area_m2"] > 0.0
        assert row["area_centroid_major_radius_m"] > 0.0
        assert row["exact_solid_of_revolution_m3"] == pytest.approx(
            row["first_moment_approximation_m3"], rel=2.0e-14
        )


def test_published_solved_volume_is_the_clipped_core_volume(receipt):
    comparison = receipt["published_volume_comparison"]
    clipping = receipt["clipping"]

    assert comparison["disagreeing_published_volume"] == "solved_confined_core_volume"
    assert clipping["confined_core_toroidal_volume_m3"] == pytest.approx(
        comparison["solved_m3"], rel=2.0e-14
    )
    assert comparison[
        "reference_published_over_own_boundary_enclosed"
    ] == pytest.approx(1.0, rel=0.03)
    assert comparison["solved_published_over_own_boundary_enclosed"] < 0.5


def test_candidate_discrimination_is_quantitative(receipt):
    candidates = receipt["candidate_discrimination"]
    ratios = receipt["controlled_ratios"]
    published = receipt["published_volume_comparison"]

    assert candidates["confined_core_clipping"]["verdict"] == "SUPPORTED"
    assert candidates["genuinely_different_curves"]["verdict"] == "EXCLUDED"
    assert candidates["revolution_or_jacobian_convention"]["verdict"] == "EXCLUDED"
    assert ratios["solved_over_stored_poloidal_area"] > 0.8
    assert ratios["solved_over_stored_boundary_enclosed_toroidal_volume"] > 0.8
    assert (
        abs(
            published["solved_over_reference"]
            - ratios["solved_over_stored_area_centroid_major_radius"]
        )
        > 0.01
    )


def test_clipping_area_and_current_fractions_are_reported_together(receipt):
    clipping = receipt["clipping"]

    assert clipping["confined_core_cell_count"] < clipping["cell_count"]
    assert clipping["confined_core_current_a"] == pytest.approx(416958.2541259555)
    assert clipping["pinned_all_domain_current_a"] == pytest.approx(933034.875)
    assert clipping["confined_core_current_fraction"] == pytest.approx(
        0.4468838896573459
    )
    assert abs(clipping["area_fraction_minus_current_fraction"]) < 0.1


def test_boundary_support_rescore_preserves_banked_means(receipt):
    rescore = receipt["boundary_support_moment_rescore"]
    beta = rescore["metrics"]["poloidal_beta"]
    internal_inductance = rescore["metrics"]["internal_inductance"]

    assert beta["original_signed_relative_deviation"] == pytest.approx(
        -0.5628443566810165
    )
    assert internal_inductance["original_signed_relative_deviation"] == pytest.approx(
        -0.7343445074393564
    )
    assert beta["deficit_eliminated"]
    assert not internal_inductance["deficit_eliminated"]
    assert rescore["parity_consequence"] == {
        "poloidal_beta_volume_deficit": "ELIMINATED",
        "internal_inductance_volume_deficit": "SURVIVES",
    }


def test_receipt_preserves_protected_evidence_and_matches_checked_in_file(receipt):
    assert receipt["execution_contract"]["nonlinear_solve_calls"] == 0
    assert receipt["protected_banked_artifacts"] == {
        "declared_count": 23,
        "verified_digest_count": 23,
        "all_digests_match": True,
        "source_and_output_receipts_are_outside_protected_set": True,
    }
    assert json.loads(OUTPUT.read_text()) == receipt
