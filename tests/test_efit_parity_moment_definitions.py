import json
from pathlib import Path

import pytest

from benchmarks.efit_parity_moment_definitions import (
    EXPECTED_REFERENCE_BETA,
    EXPECTED_REFERENCE_LI,
    rescore,
)

RECEIPT = Path(
    "docs/figures/efit-forward-parity/converged-root-geometry-attribution.json"
)


@pytest.fixture(scope="module")
def scorecard():
    return rescore(RECEIPT)


def test_reference_boundary_definition_reproduces_published_moments(scorecard):
    reproduction = scorecard["reference_definition_reproduction"]

    assert reproduction["poloidal_beta"]["reproduced"] == pytest.approx(
        EXPECTED_REFERENCE_BETA, rel=1.0e-12
    )
    assert reproduction["internal_inductance"]["reproduced"] == pytest.approx(
        EXPECTED_REFERENCE_LI, rel=1.0e-12
    )
    assert reproduction["poloidal_beta"]["passes"]
    assert reproduction["internal_inductance"]["passes"]


def test_each_definition_scores_both_integral_sides_and_current_supports(scorecard):
    rows = scorecard["four_by_two_rescore"]["rows"]

    assert len(rows) == 4
    assert {(row["definition"], row["side"]) for row in rows} == {
        ("nova_shared_volume_current", "solved"),
        ("nova_shared_volume_current", "reference"),
        ("reference_boundary_field", "solved"),
        ("reference_boundary_field", "reference"),
    }
    for row in rows:
        assert set(row["poloidal_beta"]) == {
            "confined_core",
            "all_domain_constrained",
        }
        if row["definition"] == "nova_shared_volume_current":
            assert set(row["internal_inductance"]) == {
                "confined_core",
                "all_domain_constrained",
            }
        else:
            assert set(row["internal_inductance"]) == {"current_independent"}


def test_matched_deviations_are_definition_and_support_invariant(scorecard):
    rows = scorecard["four_by_two_rescore"]["rows"]
    by_key = {(row["definition"], row["side"]): row for row in rows}
    expected_beta = scorecard["matched_definition_deviations"]["poloidal_beta"][
        "signed_relative_deviation"
    ]
    expected_li = scorecard["matched_definition_deviations"]["internal_inductance"][
        "signed_relative_deviation"
    ]

    for definition in (
        "nova_shared_volume_current",
        "reference_boundary_field",
    ):
        solved = by_key[(definition, "solved")]
        reference = by_key[(definition, "reference")]
        for support in ("confined_core", "all_domain_constrained"):
            beta_deviation = (
                solved["poloidal_beta"][support]["value"]
                / reference["poloidal_beta"][support]["value"]
                - 1.0
            )
            assert beta_deviation == pytest.approx(expected_beta, rel=1.0e-14)
        li_supports = (
            ("confined_core", "all_domain_constrained")
            if definition == "nova_shared_volume_current"
            else ("current_independent",)
        )
        for support in li_supports:
            li_deviation = (
                solved["internal_inductance"][support]["value"]
                / reference["internal_inductance"][support]["value"]
                - 1.0
            )
            assert li_deviation == pytest.approx(expected_li, rel=1.0e-14)


def test_receipt_is_table_renderable_and_preserves_banked_evidence(scorecard):
    table = scorecard["four_by_two_rescore"]["html_table"]
    integrity = scorecard["protected_banked_artifacts"]

    assert table.startswith('<div style="overflow-x:auto"><table>')
    assert table.count("<tr>") == 5
    assert table.endswith("</table></div>")
    assert integrity == {
        "declared_count": 23,
        "verified_digest_count": 23,
        "all_digests_match": True,
        "source_receipt_and_this_receipt_are_outside_protected_set": True,
    }


def test_checked_in_receipt_matches_the_arithmetic(scorecard):
    output = Path("docs/figures/efit-forward-parity/moment-definition-rescore.json")
    assert json.loads(output.read_text()) == scorecard
