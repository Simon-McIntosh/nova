"""Dimension-independent contracts for projected Krylov conditioning."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import newton_krylov


def _diagonal_fixed_point(condition: float, dimension: int):
    """Return a linear map with a logarithmically distributed action spectrum."""
    diagonal = jnp.exp(jnp.linspace(0.0, -jnp.log(condition), dimension))
    action = jnp.diag(diagonal)
    tangent = jnp.eye(dimension) - action
    offset = jnp.ones(dimension)
    return lambda state: tangent @ state + offset


def _solve(condition: float, dimension: int, *, ratio_limit: float = math.e):
    """Run one fixed-shape Newton step at a requested projection dimension."""
    return newton_krylov(
        _diagonal_fixed_point(condition, dimension),
        jnp.zeros(dimension),
        newton_steps=1,
        gmres_iterations=dimension,
        warmup=0,
        step_cap=1.0e6,
        krylov_condition_limit=ratio_limit,
    )


@pytest.mark.parametrize("dimension", [8, 12, 24])
def test_spectral_ratio_engages_without_dimension_calibration(dimension: int):
    """A separated singular tail engages at every requested dimension."""
    conditioned = _solve(200.0, dimension)
    control = _solve(200.0, dimension, ratio_limit=jnp.inf)

    assert int(conditioned.krylov_conditioning_count) == 1
    np.testing.assert_allclose(
        float(conditioned.maximum_projected_krylov_condition), 200.0, rtol=3.0e-6
    )
    assert np.max(np.abs(np.asarray(conditioned.state))) < np.max(
        np.abs(np.asarray(control.state))
    )


@pytest.mark.parametrize("dimension", [8, 12, 24])
def test_typical_spectrum_is_not_damped_at_any_dimension(dimension: int):
    """An order-one condition spread remains unchanged at every dimension."""
    conditioned = _solve(4.0, dimension)
    control = _solve(4.0, dimension, ratio_limit=jnp.inf)

    assert int(conditioned.krylov_conditioning_count) == 0
    np.testing.assert_array_equal(conditioned.state, control.state)


@pytest.mark.parametrize("dimension", [8, 12, 24])
def test_conditioning_remains_jit_and_vmap_safe(dimension: int):
    """Batched compiled solves retain fixed result shapes and engagement receipts."""

    def solve(condition):
        return _solve(condition, dimension)

    result = jax.jit(jax.vmap(solve))(jnp.asarray([4.0, 200.0]))
    np.testing.assert_array_equal(result.krylov_conditioning_count, [0, 1])
    assert result.state.shape == (2, dimension)


def test_dataset_fitted_conditioning_constants_are_absent():
    """The solver path contains neither retired trajectory calibration value."""
    source = Path("nova/equilibrium/fixed_point.py").read_text(encoding="utf-8")
    assert "44.5" not in source
    assert "27.781718445022726" not in source
    assert "_PROJECTED_KRYLOV_CONDITION_DIMENSION" not in source
