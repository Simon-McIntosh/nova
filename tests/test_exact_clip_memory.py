"""Focused contracts for exact-clip memory attribution."""

import pytest

from benchmarks.exact_clip_memory_scaling import scaling_exponent


def test_scaling_exponent_identifies_linear_and_pairwise_growth():
    """Power-law attribution distinguishes per-cell from all-cell pairs."""
    assert scaling_exponent(100, 300, 10, 30) == pytest.approx(1.0)
    assert scaling_exponent(100, 900, 10, 30) == pytest.approx(2.0)


@pytest.mark.parametrize("value", [0, -1])
def test_scaling_exponent_refuses_nonpositive_measurements(value):
    """An empty or signed measurement cannot support a memory-law claim."""
    with pytest.raises(ValueError, match="positive"):
        scaling_exponent(value, 300, 10, 30)
