"""Roundoff policy for independently evaluated terminal class margins."""

import numpy as np
import pytest

from benchmarks.diiid_forward_gs_match import (
    _CLASS_MARGIN_ROUNDOFF_ATOL,
    _CLASS_MARGIN_ROUNDOFF_RTOL,
    _class_margins_agree_within_roundoff,
)


def test_observed_reduction_order_difference_is_within_roundoff_bound():
    """The measured sixteen-ULP re-read remains the same terminal operand."""

    terminal = float.fromhex("0x1.cf8f16d4adaa0p-6")
    diagnostic = float.fromhex("0x1.cf8f16d4ada90p-6")

    assert abs(diagnostic - terminal) == 5.551115123125783e-17
    assert _class_margins_agree_within_roundoff(diagnostic, terminal)


def test_material_margin_change_exceeds_named_roundoff_bound():
    """The guard still rejects a change outside both declared tolerances."""

    terminal = 0.028293392463377587
    bound = _CLASS_MARGIN_ROUNDOFF_ATOL + _CLASS_MARGIN_ROUNDOFF_RTOL * abs(terminal)

    assert not _class_margins_agree_within_roundoff(terminal + 2.0 * bound, terminal)
    assert _CLASS_MARGIN_ROUNDOFF_RTOL == pytest.approx(1.0e-12)
    assert _CLASS_MARGIN_ROUNDOFF_ATOL == np.finfo(np.float64).eps
