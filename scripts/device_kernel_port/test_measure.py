"""Focused checks for the packed flux-kernel receipt machinery."""

from __future__ import annotations

import numpy as np

from scripts.device_kernel_port.measure import _ordered_bits, ulp_distribution


def test_ordered_bits_increase_across_sign_and_zero() -> None:
    values = np.asarray([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0])
    ordered = _ordered_bits(values)
    assert np.all(ordered[1:] >= ordered[:-1])


def test_ulp_distribution_preserves_every_distance() -> None:
    expected = np.asarray([1.0, 1.0, -1.0, 0.0])
    actual = expected.copy()
    actual[1] = np.nextafter(actual[1], np.inf)
    actual[2] = np.nextafter(np.nextafter(actual[2], -np.inf), -np.inf)
    distribution = ulp_distribution(actual, expected)
    assert distribution["histogram"] == {"0": 2, "1": 1, "2": 1}
    assert distribution["byte_identical_fraction"] == 0.5
    assert distribution["max"] == 2


def test_ulp_distribution_refuses_nonfinite_values() -> None:
    with np.testing.assert_raises_regex(ValueError, "finite"):
        ulp_distribution(np.asarray([np.inf]), np.asarray([1.0]))
