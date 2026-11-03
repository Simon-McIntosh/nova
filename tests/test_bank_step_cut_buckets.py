"""Cut-cell booking evaluates only the selected branch at static shape."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, support_read, topology
from nova.jax.config import configure_dtypes
from tests.test_support_read import UnitDensity, _case

configure_dtypes()


def test_normal_cut_count_over_batch_capacity():
    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    reading = support.reading
    cut = np.flatnonzero(
        (np.asarray(reading.membership) > 0)
        & (np.asarray(reading.membership) < 1 - 1e-14)
    )
    capacity = current._NORMAL_CUT_BATCH_WIDTH
    assert len(cut) > capacity
    live_normal = cut[np.asarray(reading.normal_form_cells)[cut]]
    assert len(live_normal) > 0
    overflow_cell = live_normal[0]
    chosen = np.concatenate((cut[cut != overflow_cell][:capacity], [overflow_cell]))
    flags = jnp.zeros_like(reading.normal_form_cells).at[chosen].set(True)
    reading = reading._replace(normal_form_cells=flags)
    normal_count = int(jnp.count_nonzero(flags))
    assert normal_count == capacity + 1
    assert (normal_count + capacity - 1) // capacity > 1

    psi = support_read.nodal_jets(
        operands, tuple(jnp.zeros_like(booked.cell_current) for _ in range(3))
    )[operands.sample_index, 0]
    direction = jnp.ones_like(psi)
    field_direction = jnp.zeros_like(reading.field_coefficients)
    quadrature = current.Quadrature(
        sigma=topology.TopologyConvention.from_cocos(17, 1.0).sigma, segments=8
    )

    def gradient(radius, flux):
        return jnp.ones_like(radius)

    expected = current.book_tangent(
        psi,
        direction,
        reading,
        field_direction,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        operands.geometry,
        UnitDensity(),
        gradient,
        quadrature,
        booked,
    )

    def batched(flux, local_read):
        return current.book_tangent_channels(
            flux,
            direction[None],
            local_read,
            field_direction[None],
            jnp.asarray([0.0]),
            jnp.asarray([0.0]),
            operands.geometry,
            UnitDensity(),
            gradient,
            quadrature,
            booked,
            jnp.asarray([0]),
        )

    observed = jax.jit(batched)(psi, reading)
    actual_values = np.asarray(observed[..., 0])
    expected_values = np.stack(tuple(map(np.asarray, expected)), axis=-1)
    assert chosen[-1] == overflow_cell
    assert bool(np.asarray(support.reading.normal_form_cells)[overflow_cell])
    booked_current = np.asarray(booked.cell_current)[overflow_cell]
    assert np.isfinite(booked_current)
    assert booked_current != 0
    assert np.isfinite(expected_values[overflow_cell]).all()
    assert np.isfinite(actual_values[overflow_cell]).all()
    np.testing.assert_allclose(
        actual_values[overflow_cell],
        expected_values[overflow_cell],
        rtol=1e-10,
        atol=1e-10,
    )


def test_cell_vmap_has_no_per_cell_branch():
    source = inspect.getsource(current.book_tangent_channels)
    assert "jax.vmap(cut_cell)" not in source
