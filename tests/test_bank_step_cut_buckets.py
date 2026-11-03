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
    assert len(cut) > current._NORMAL_CUT_BATCH_WIDTH
    chosen = cut[: current._NORMAL_CUT_BATCH_WIDTH + 1]
    flags = jnp.zeros_like(reading.normal_form_cells).at[chosen].set(True)
    reading = reading._replace(normal_form_cells=flags)
    assert int(jnp.count_nonzero(flags)) > current._NORMAL_CUT_BATCH_WIDTH

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
    observed = jax.jit(
        lambda flux, local_read: current.book_tangent_channels(
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
    )(psi, reading)
    np.testing.assert_allclose(
        np.asarray(observed[..., 0]),
        np.stack(tuple(map(np.asarray, expected)), axis=-1),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )


def test_branch_selection_is_outside_cell_vmap():
    source = inspect.getsource(current.book_tangent_channels)
    assert "jax.vmap(cut_cell)" not in source
