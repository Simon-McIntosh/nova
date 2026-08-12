"""Synthetic recovery tests for partial regression."""

import numpy as np
import pytest

from nova.calibrate.partial import PartialRegressionError, partial_fit


def test_difference_coefficient_is_recovered_after_level_drives_are_removed():
    generator = np.random.default_rng(17)
    samples = 1200
    level = generator.normal(size=samples)
    nuisance = generator.normal(size=samples)
    difference = 0.8 * level + generator.normal(scale=0.2, size=samples)
    target = 4.25 * difference - 7.0 * level + 1.7 * nuisance + 3.0

    fit = partial_fit(target, difference, np.column_stack((level, nuisance)))

    assert fit.slope == pytest.approx(4.25, rel=1.0e-12)
    assert fit.variance_explained == pytest.approx(1.0, abs=1.0e-12)
    assert fit.residual == pytest.approx(0.0, abs=1.0e-12)


def test_injected_scatter_sets_the_recovery_tolerance():
    generator = np.random.default_rng(29)
    controls = generator.normal(size=(4000, 3))
    regressor = controls[:, 0] + generator.normal(size=4000)
    target = 0.37 * regressor + controls @ np.array([2.0, -1.0, 0.5])
    target += generator.normal(scale=0.04, size=target.size)

    fit = partial_fit(target, regressor, controls)

    assert fit.slope == pytest.approx(0.37, abs=2.0e-3)
    assert fit.residual == pytest.approx(0.04, rel=0.08)


def test_a_regressor_entirely_inside_the_control_span_is_not_identified():
    control = np.linspace(-1.0, 1.0, 200)
    with pytest.raises(PartialRegressionError, match="outside the controls"):
        partial_fit(2.0 * control, control, control)
