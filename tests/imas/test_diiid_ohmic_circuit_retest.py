"""Tests for the DIII-D omitted-ohmic circuit retest."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_ohmic_circuit_retest.py"
SPEC = importlib.util.spec_from_file_location("diiid_ohmic_circuit_retest", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RECEIPT_PATH = (
    Path(__file__).parents[2]
    / "docs/figures/diiid-forward-onboarding/ohmic-circuit"
    / MODULE.RECEIPT_NAME
)


def test_signed_scale_recovers_gain_and_polarity_without_offset():
    reference = np.array([-4.0, -1.0, 2.0, 5.0])
    target = -2.5 * reference
    positive = MODULE.relationship_metrics(reference, target, positive_scale_only=True)
    signed = MODULE.relationship_metrics(reference, target)
    assert positive["scale_target_from_ecoila"] == 0.0
    assert signed["scale_target_from_ecoila"] == -2.5
    assert signed["relative_l2_residual"] == 0.0
    assert signed["maximum_absolute_residual"] == 0.0


def test_turn_bases_preserve_equivalent_current_residual():
    result = MODULE.evaluate_coil(
        np.array([1.0, 2.0, 4.0]),
        np.array([1.1, 2.2, 4.3]),
        reference_turns=48.0,
        target_turns=6.0,
    )
    current = result["current_A_basis"][
        "family_gain_and_polarity_accounted_signed_scale"
    ]
    per_turn = result["current_per_declared_turn_basis"][
        "family_gain_and_polarity_accounted_signed_scale"
    ]
    ampere_turn = result["ampere_turn_basis"][
        "family_gain_and_polarity_accounted_signed_scale"
    ]
    assert np.isclose(
        current["maximum_equivalent_current_residual_A"],
        per_turn["maximum_equivalent_current_residual_A"],
    )
    assert np.isclose(
        current["maximum_equivalent_current_residual_A"],
        ampere_turn["maximum_equivalent_current_residual_A"],
    )
    assert np.isclose(
        per_turn["scale_target_from_ecoila"],
        8.0 * current["scale_target_from_ecoila"],
    )
    assert np.isclose(
        ampere_turn["scale_target_from_ecoila"],
        current["scale_target_from_ecoila"] / 8.0,
    )


def test_physical_bound_is_one_percent_of_measured_ohmic_peak():
    assert MODULE.DETERMINISTIC_MAXIMUM_RESIDUAL_PERCENT == 1.0
    assert np.isclose(
        MODULE.OHMIC_PEAK_REFERENCE_A / 100.0,
        554.83703125,
    )


def test_published_receipt_reports_all_samples_and_subset_verdict():
    receipt = json.loads(RECEIPT_PATH.read_text())
    assert receipt["common_time_base"]["sample_count"] == 480_256
    assert set(receipt["per_coil"]) == set(MODULE.TARGET_COILS)
    assert receipt["summary"]["deterministic_coils"] == ["E567UP", "E89DN"]
    assert receipt["summary"]["non_deterministic_coils"] == [
        "ECOILB",
        "E567DN",
        "E89UP",
    ]
    assert receipt["summary"]["all_five_recoverable_from_ecoila"] is False
    for result in receipt["per_coil"].values():
        assert result["fault_accounting"]["time_local_freedom"] is False
        assert result["fault_accounting"]["offset_fitted"] is False
        assert result["fault_accounting"]["polarity"] == 1
        common_scale = result["fault_accounting"]["common_family_multiplier_check"]
        assert abs(common_scale["relative_l2_difference_from_unscaled"]) < 1.0e-14
        assert np.isclose(
            common_scale["maximum_equivalent_current_residual_A"],
            result["verdict"]["observed_maximum_residual_A"],
        )
