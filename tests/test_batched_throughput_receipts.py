"""Receipts and continuity contract of the batched forward budget."""

import numpy as np
import pytest

from benchmarks.diiid_batched_throughput import (
    BranchReceipt,
    validate_warm_start_continuity,
)
from nova.equilibrium.diagnostics import vertical_conditioning_receipt


def _branch(identity: str, axis_r: float) -> BranchReceipt:
    return BranchReceipt(
        identity=identity,
        diverted=identity == "diverted_plasma",
        plasma_current_a=8.0e5 if identity != "vacuum" else 0.0,
        axis_r_m=axis_r,
        axis_z_m=0.02,
    )


def test_vertical_conditioning_receipt_carries_decay_index_and_margin():
    radius = np.linspace(0.6, 1.4, 2001)
    vertical_field = -0.2 / radius
    receipt = vertical_conditioning_receipt(radius, vertical_field, 1.0)
    assert receipt.decay_index == pytest.approx(1.0, abs=2.0e-6)
    assert receipt.lower_stability_margin == pytest.approx(1.0, abs=2.0e-6)
    assert receipt.upper_stability_margin == pytest.approx(0.5, abs=2.0e-6)
    assert receipt.stability_margin == pytest.approx(0.5, abs=2.0e-6)
    assert receipt.stable


def test_warm_start_continuity_accepts_one_shot_on_one_branch():
    first = [_branch("diverted_plasma", 1.00), _branch("diverted_plasma", 1.01)]
    second = [_branch("diverted_plasma", 1.02), _branch("diverted_plasma", 1.03)]
    validate_warm_start_continuity(
        "shot-alpha", first, second, maximum_axis_step_m=0.05
    )


def test_warm_start_continuity_refuses_a_silent_branch_change():
    first = [_branch("diverted_plasma", 1.00), _branch("diverted_plasma", 1.01)]
    changed = [_branch("diverted_plasma", 1.02), _branch("vacuum", 1.01)]
    with pytest.raises(RuntimeError, match="member 1 changed branch"):
        validate_warm_start_continuity(
            "shot-alpha", first, changed, maximum_axis_step_m=0.05
        )


def test_warm_start_continuity_refuses_an_axis_jump_on_the_same_branch():
    first = [_branch("limited_plasma", 1.00)]
    jumped = [_branch("limited_plasma", 1.12)]
    with pytest.raises(RuntimeError, match="axis moved"):
        validate_warm_start_continuity(
            "shot-alpha", first, jumped, maximum_axis_step_m=0.05
        )


def test_warm_start_continuity_refuses_a_nonfinite_axis_receipt():
    first = [_branch("limited_plasma", 1.00)]
    invalid = [_branch("limited_plasma", float("nan"))]
    with pytest.raises(RuntimeError, match="non-finite branch axis"):
        validate_warm_start_continuity(
            "shot-alpha", first, invalid, maximum_axis_step_m=0.05
        )
