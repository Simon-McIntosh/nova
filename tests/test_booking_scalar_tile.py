"""Scalar conic populations retain every moment across complete and partial tiles."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, support_read
from tests.test_support_read import UnitDensity, _case

assert jax.config.jax_enable_x64 is True


class CellGeometry(NamedTuple):
    vertices: object
    vertex_count: object
    centre: object
    pitch: object
    sample_points: object
    fit_inverse: object


def _unbucketed(mask, width, evaluate, zero):
    """Independent placement oracle: integrate every cell and mask its result."""
    values = jax.vmap(evaluate)(jnp.arange(len(mask)))
    return jax.tree.map(lambda value: jnp.where(mask, value, 0.0), values)


@pytest.fixture(scope="module", params=["limited", "diverted"])
def physical(request):
    operands, cache, roots, pitch, step, *_ = _case(request.param)
    support, booked = jax.block_until_ready(
        step(operands, cache, roots + 0.03 * pitch, roots)
    )
    assert bool(support.valid) and bool(booked.valid)
    psi = support_read.nodal_jets(operands, (jnp.zeros(1),) * 3)[
        operands.sample_index, 0
    ]
    return operands, support, psi


def _population(physical, count):
    operands, support, psi = physical
    reading = support.reading
    conic = np.flatnonzero(
        (np.asarray(reading.membership) > 0)
        & (np.asarray(reading.membership) < 1 - 1e-14)
        & ~np.asarray(reading.normal_form_cells)
    )
    assert len(conic) > 0
    # Repeated physical cut cells exercise population scheduling independently
    # of mesh resolution, without changing any cell's local flux or geometry.
    indices = jnp.asarray(np.resize(conic, count))
    geometry = CellGeometry(
        *(getattr(operands.geometry, name)[indices] for name in CellGeometry._fields)
    )
    reading = reading._replace(
        **{
            name: getattr(reading, name)[indices]
            for name in (
                "membership",
                "field_coefficients",
                "fragment_selected",
                "normal_form_cells",
                "edge_interval",
            )
        }
    )
    return psi[indices], reading, geometry


def _assert_booking(actual, expected):
    assert bool(actual.valid) and bool(expected.valid)
    assert float(expected.raw_current) > 0
    for name in actual._fields[:-1]:
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=1e-12,
            atol=1e-14,
            err_msg=name,
        )


@pytest.mark.parametrize("count", [17, 47, 67, 137])
def test_all_booked_moments_match_independent_population(physical, count):
    psi, reading, geometry = _population(physical, count)
    rule = current.Quadrature(segments=8, target_current=3.7)
    with patch.object(current, "_integrate_bucket", _unbucketed):
        expected = jax.jit(current.book)(psi, reading, geometry, UnitDensity(), rule)
    actual = jax.jit(lambda *a: current.book(*a))(
        psi, reading, geometry, UnitDensity(), rule
    )
    _assert_booking(actual, expected)
    with patch.object(current, "_SCALAR_CUT_BATCH_WIDTH", 32):
        base = jax.jit(lambda *a: current.book(*a))(
            psi, reading, geometry, UnitDensity(), rule
        )
    _assert_booking(actual, base)
    print("POPULATION", count, "RAW", float(actual.raw_current), flush=True)


def test_scalar_tile_is_wide_and_normal_tile_is_independent():
    assert current._SCALAR_CUT_BATCH_WIDTH == 64
    assert current._NORMAL_CUT_BATCH_WIDTH == 8


@pytest.mark.parametrize("count", [0, 17, 67, 128, 137, 257])
def test_bucket_bound_includes_partial_population(count):
    cells = 270
    mask = jnp.arange(cells) < count
    result = jax.jit(
        lambda mask: current._integrate_bucket(
            mask, 128, lambda index: (index + 1.0) ** 2, jnp.zeros(cells)
        )
    )(mask)
    expected = np.where(np.arange(cells) < count, (np.arange(cells) + 1.0) ** 2, 0)
    np.testing.assert_array_equal(result, expected)


def test_trial_and_converged_fresh_residuals_match_base(physical):
    operands, support, _ = physical
    cells = len(operands.geometry.centre)
    operands = replace(operands, rows=jnp.zeros((len(operands.nodes), cells, 3, 6)))
    zero = (jnp.zeros(cells),) * 3
    rule = current.Quadrature(segments=8)

    def residual(bank, state, reading):
        return support_read.step_residual(
            bank, state, state, reading, UnitDensity(), rule
        )

    with patch.object(current, "_SCALAR_CUT_BATCH_WIDTH", 32):
        base_program = jax.jit(residual).lower(operands, zero, support).compile()
    head_program = (
        jax.jit(lambda *args: residual(*args)).lower(operands, zero, support).compile()
    )
    converged = base_program(operands, zero, support).booked[:3]
    for label, state in (("trial", zero), ("converged", converged)):
        base = base_program(operands, state, support)
        head = head_program(operands, state, support)
        _assert_booking(head.booked, base.booked)
        for actual, expected in zip(head.residual, base.residual, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        norm = float(jnp.linalg.norm(jnp.concatenate(head.residual)))
        if label == "converged":
            assert norm <= 1e-10
        print("FRESH_RESIDUAL", label, norm, flush=True)
