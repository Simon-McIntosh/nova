"""Tests for the DIII-D producer-current family-scale receipt."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_producer_current_scaling.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_producer_current_scaling", MODULE_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RECEIPT_PATH = (
    Path(__file__).parents[2]
    / "docs/figures/diiid-forward-onboarding/producer-currents/"
    / MODULE.RECEIPT_NAME
)


def test_single_scalar_fit_has_no_offset_or_per_channel_freedom():
    result = MODULE.fit_single_positive_scale(
        reference=np.array([2.0, 4.0, 8.0]),
        candidate=np.array([1.0, 2.0, 4.0]),
    )
    assert result["scale"] == 2.0
    assert result["relative_l2_residual"] == 0.0
    assert result["explained_energy_fraction"] == 1.0


def test_boundary_diagnostic_removes_only_an_additive_gauge():
    result = MODULE.boundary_misclosure(
        ohmic_flux=[np.array([10.0, 11.0, 12.0])],
        shaping_flux=[np.array([1.0, 0.0, -1.0])],
        axis_to_boundary_span=np.array([2.0]),
        scale=1.0,
    )
    assert result["rms_wb_total"]["median"] == 0.0
    assert result["fraction_of_axis_to_boundary_flux_span"]["median"] == 0.0


def test_published_receipt_recomputes_ratios_and_retains_residual():
    receipt = json.loads(RECEIPT_PATH.read_text())
    ratios = receipt["family_ratios"]
    assert np.isclose(ratios["raw_max_abs_current_ratio"], 595.49, atol=0.01)
    assert np.isclose(
        ratios["turns_corrected_max_abs_ampere_turn_ratio"], 513.60, atol=0.01
    )
    reconciliation = receipt["single_scalar_reconciliation"]
    assert reconciliation["per_coil_freedom"] is False
    assert reconciliation["effective_turns_applied_before_fit"] is True
    assert reconciliation["sample_count"] == 480_256
    assert 0.0 <= reconciliation["best_fit"]["relative_l2_residual"] <= 1.0


def test_vacuum_diagnostic_is_firewalled_from_solve_authority():
    receipt = json.loads(RECEIPT_PATH.read_text())
    diagnostic = receipt["vacuum_boundary_internal_diagnostic"]
    assert diagnostic["role"] == "internal_diagnostic_only"
    assert diagnostic["slice_count"] == 340
    assert diagnostic["global_current_sign_invariant"] is True
    assert diagnostic["confidence"] == "low"
    assert receipt["solve_authority"] == {
        "may_drive_a_solve": False,
        "verdict": "prohibited_as_solve_input",
        "applies_to": "the netCDF producer currents as stored or benchmark-scaled",
        "reason": receipt["solve_authority"]["reason"],
    }
