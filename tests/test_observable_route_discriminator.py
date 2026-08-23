from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks import observable_route_discriminator as discriminator


def _case(
    shot: int,
    state_difference: float,
    differences: dict[str, tuple[float, float]],
):
    observables = {}
    for observable, (route_difference, shared_difference) in differences.items():
        localisation = None
        if shared_difference > 0.0:
            localisation = {
                "operation": "observe_moments.volume_reduction",
                "maximum_absolute_difference": shared_difference,
            }
        observables[observable] = {
            "route_observable_difference": {
                "maximum_absolute_difference": route_difference,
            },
            "shared_state_evaluation": {
                "maximum_absolute_difference": shared_difference,
            },
            "first_structurally_differing_operation": localisation,
        }
    return {
        "shot": shot,
        "slice_index": shot % 100,
        "terminal_state_difference": {
            "maximum_relative_difference": state_difference,
        },
        "observables": observables,
    }


def test_difference_retains_exact_and_scale_normalised_results():
    equal = discriminator._difference(np.array([1.0]), np.array([1.0]))
    changed = discriminator._difference(np.array([2.0]), np.array([2.5]))

    assert equal == {
        "exactly_equal": True,
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
    }
    assert changed["exactly_equal"] is False
    assert changed["maximum_absolute_difference"] == pytest.approx(0.5)
    assert changed["maximum_relative_difference"] == pytest.approx(0.25)


def test_correlations_cover_six_cases_and_retain_tied_ranks():
    result = discriminator._correlations(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
    )

    assert result["case_count"] == 6
    assert result["pearson_r"] == pytest.approx(0.9389529557)
    assert result["spearman_rho"] == pytest.approx(0.9856107606)
    assert result["interpretation"] == "finite correlation across all six cases"


def test_verdict_requires_a_shared_state_difference():
    observable = discriminator.TARGET_OBSERVABLES[0]
    cases = [
        _case(
            shot,
            state_difference=float(index + 1),
            differences={observable: (float(index), 1.0e-16 if index == 2 else 0.0)},
        )
        for index, shot in enumerate((21978, 21983, 21985, 21986, 21989, 22086))
    ]

    result = discriminator._observable_receipt(observable, cases)

    assert result["verdict"] == "COMPUTATION_DIFFERS"
    assert result["shared_state_case_count"] == 6
    assert result["shared_state_maximum_absolute_difference"] == pytest.approx(1.0e-16)
    assert (
        result["first_structurally_differing_operation"]
        == "observe_moments.volume_reduction"
    )
    assert len(result["difference_against_state_agreement"]["cases"]) == 6


def test_exact_shared_state_agreement_assigns_state_inheritance():
    observable = discriminator.TARGET_OBSERVABLES[1]
    cases = [
        _case(
            shot,
            state_difference=float(index + 1),
            differences={observable: (float(index), 0.0)},
        )
        for index, shot in enumerate((21978, 21983, 21985, 21986, 21989, 22086))
    ]

    result = discriminator._observable_receipt(observable, cases)

    assert result["verdict"] == "STATE_INHERITED"
    assert result["first_structurally_differing_operation"] is None
    assert result["localisation_by_differing_case"] == []


def test_committed_receipt_carries_complete_discriminator_evidence():
    receipt = json.loads(discriminator.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    observables = {row["observable"]: row for row in receipt["observables"]}

    alignment = receipt["measurement_contract"]["backend_alignment"]
    expected_status = (
        "complete" if alignment["matches"] else "provisional_backend_mismatch"
    )
    assert receipt["status"] == expected_status
    assert receipt["measurement_contract"]["case_count"] == 6
    assert receipt["measurement_contract"]["no_repair_attempted"] is True
    assert set(observables) == set(discriminator.TARGET_OBSERVABLES)
    assert sum(receipt["verdict_counts"].values()) == 3
    platform_measurements = receipt["platform_measurements"]
    comparisons = {
        row["observable"]: row
        for row in receipt["cross_platform_observable_comparison"]
    }
    assert set(comparisons) == set(discriminator.TARGET_OBSERVABLES)
    assert "cpu" in platform_measurements
    if receipt["status"] == "complete":
        assert receipt["backend"]["platform"] in platform_measurements
        assert len(platform_measurements) >= 2
    volume_evidence = comparisons["moments.volume"]["exact_equality_bound_evidence"]
    assert volume_evidence["cpu_shared_state_difference_is_exactly_zero"] is True
    assert volume_evidence["shared_state_difference_by_platform"]["cpu"] == 0.0
    if receipt["status"] == "complete":
        assert (
            observables["moments.volume"]["first_structurally_differing_operation"]
            == "observe_moments.volume_reduction"
        )
    for row in observables.values():
        assert row["verdict"] in {"COMPUTATION_DIFFERS", "STATE_INHERITED"}
        assert len(row["shared_state_evaluations"]) == 6
        assert len(row["difference_against_state_agreement"]["cases"]) == 6
        correlation = row["difference_against_state_agreement"]["correlation"]
        assert correlation["case_count"] == 6
        for evaluation in row["shared_state_evaluations"]:
            assert "scalar_forward_profile_observe" in evaluation
            assert "jitted_vmap_forward_profile_observe" in evaluation
            assert "maximum_absolute_difference" in evaluation
        if row["verdict"] == "COMPUTATION_DIFFERS":
            assert row["first_structurally_differing_operation"]
        else:
            assert row["first_structurally_differing_operation"] is None
