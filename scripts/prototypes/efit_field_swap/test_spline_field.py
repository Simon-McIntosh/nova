"""Check the adapter's spline evaluator against scipy's own bispline."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline, bisplev

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record

CAPTURE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission/traces/jt60sa-5slice/"
    "boundary_slice_5_iter_301.txt"
)


@pytest.fixture(scope="module")
def record():
    return read_trace_record(CAPTURE)


def test_value_and_derivatives_match_scipy(record):
    field = TraceSplineField.from_record(record)
    tx, tz, c = RectBivariateSpline(
        record.rgrid, record.zgrid, record.psi, kx=3, ky=3, s=0
    ).tck
    tck = (tx, tz, c, 3, 3)
    rng = np.random.default_rng(7)
    r = rng.uniform(record.rgrid[2], record.rgrid[-3], 64)
    z = rng.uniform(record.zgrid[2], record.zgrid[-3], 64)
    for ri, zi in zip(r, z, strict=True):
        jet = field.evaluate(np.asarray((ri, zi)))
        value = float(bisplev(ri, zi, tck))
        gr = float(bisplev(ri, zi, tck, dx=1))
        gz = float(bisplev(ri, zi, tck, dy=1))
        hrr = float(bisplev(ri, zi, tck, dx=2))
        hrz = float(bisplev(ri, zi, tck, dx=1, dy=1))
        hzz = float(bisplev(ri, zi, tck, dy=2))
        scale = max(abs(value), 1e-6)
        assert abs(float(jet.value) - value) <= 1e-9 * scale
        assert abs(float(jet.gradient[0]) - gr) <= 1e-7 * max(abs(gr), 1.0)
        assert abs(float(jet.gradient[1]) - gz) <= 1e-7 * max(abs(gz), 1.0)
        assert abs(float(jet.hessian[0, 0]) - hrr) <= 1e-6 * max(abs(hrr), 1.0)
        assert abs(float(jet.hessian[0, 1]) - hrz) <= 1e-6 * max(abs(hrz), 1.0)
        assert abs(float(jet.hessian[1, 1]) - hzz) <= 1e-6 * max(abs(hzz), 1.0)


def test_evaluator_is_jax_differentiable_and_vmap_able(record):
    field = TraceSplineField.from_record(record)
    point = np.asarray((record.axis_r, record.axis_z))
    gradient = jax.grad(lambda p: field.value(p))(jnp.asarray(point))
    assert np.all(np.isfinite(np.asarray(gradient)))
    points = jnp.asarray(
        np.stack((record.rgrid[10:15], record.zgrid[10:15]), axis=1)
    )
    values = jax.vmap(field.value)(points)
    assert np.all(np.isfinite(np.asarray(values)))
    hessian = jax.jacfwd(lambda p: field.evaluate(p).hessian)(jnp.asarray(point))
    assert np.all(np.isfinite(np.asarray(hessian)))