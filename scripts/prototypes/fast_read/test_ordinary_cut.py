"""Focused checks for four-crossing quadratic Bell cuts."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize(
    ("boundary_flux", "components", "expected_fraction"),
    [
        pytest.param(0.1, 2, 0.45 + 0.05 * math.log(0.1), id="separate"),
        pytest.param(-0.1, 1, 0.55 - 0.05 * math.log(0.1), id="connected"),
    ],
)
def test_four_crossing_cut(boundary_flux, components, expected_fraction):
    """Keep the two Bell lobes distinct and preserve their booked area."""
    assert jax.config.jax_enable_x64 is True
    from scripts.prototypes.fast_read.ordinary_cut import (
        _split_polygons,
        bell_fraction,
    )

    vertices = jnp.asarray([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]])
    geometry = SimpleNamespace(
        vertices=vertices,
        vertex_count=jnp.asarray([4]),
        centre=jnp.asarray([[0.0, 0.0]]),
        pitch=jnp.asarray([1.0]),
        full_area=jnp.asarray([4.0]),
    )
    reading = SimpleNamespace(
        field_coefficients=jnp.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        boundary_flux=jnp.asarray(boundary_flux),
    )
    _, counts, cut_valid, four = _split_polygons(geometry, reading, 1.0)
    fraction, fraction_valid = bell_fraction(geometry, reading, 1.0)

    assert bool(four[0])
    assert bool(cut_valid[0] and fraction_valid[0])
    assert int(jnp.count_nonzero(counts[0])) == components
    assert abs(float(fraction[0]) - expected_fraction) < 0.006
