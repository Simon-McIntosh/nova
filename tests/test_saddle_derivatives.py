"""Saddle curvature agrees with an extended-precision bank contraction."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from decimal import localcontext
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.bank_saddle_derivatives.inspect import (
    decimal,
    exact_coefficients,
    exact_derivatives,
)


@pytest.fixture(scope="module")
def saddle_inputs():
    from tests.equilibrium import test_topology_read as fixture

    assert jax.config.jax_enable_x64
    if os.environ.get("NOVA_SADDLE_CERTIFICATE"):
        from scripts.prototypes.bank_solve_profile.fixture import build

        batch = build(16)
        archive = np.load(os.environ["NOVA_SADDLE_STATES"])
        converged = tuple(jnp.asarray(archive[f"moment_{i}"][12]) for i in range(3))
        return batch.operands, batch.cache, batch.roots, (batch.moments, converged)
    _, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    geometry = fixture._realised_hex_geometry(wall, 50)
    nodes = np.unique(
        np.round(np.asarray(geometry.sample_points).reshape(-1, 2), 12), axis=0
    )
    jets = fixed_points.stack_jet(jax.vmap(analytic.evaluate)(jnp.asarray(nodes)))
    # Large smooth source and exterior contributions cancel before curvature is read.
    rows = jnp.zeros((len(nodes), 1, 3, 6), dtype=jnp.float64)
    rows = rows.at[:, 0, 0, :].set(jets + jnp.asarray([1.0, 0.3, -0.2, 0.1, 0.2, -0.1]))
    exterior = -jnp.broadcast_to(
        jnp.asarray([1.0, 0.3, -0.2, 0.1, 0.2, -0.1]), jets.shape
    )
    bank = support_read.SupportOperands.from_rows(geometry, rows, exterior, nodes=nodes)
    cache = fixed_points.RowCache(
        bank.mesh, bank.nodes, jnp.arange(len(nodes)), rows, exterior
    )
    moments = (jnp.ones(1), jnp.zeros(1), jnp.zeros(1))
    return bank, cache, jnp.asarray([axis, saddle]), (moments, moments)


def reference(bank, moments, point):
    locator = fixed_points._CachedField(
        bank.mesh,
        jnp.arange(len(bank.nodes)),
        bank.rows,
        jnp.stack(moments, axis=-1),
        bank.exterior,
    )
    cell, triangle, slots = map(np.asarray, locator._location(point))
    rows = np.asarray(bank.rows[slots]).reshape(3, -1, 6)
    exterior = np.asarray(bank.exterior[slots])
    with localcontext() as context:
        context.prec = 80
        flat = [decimal(v) for v in np.stack(moments, axis=-1).reshape(-1)]
        jets = [
            [
                sum(decimal(a) * b for a, b in zip(rows[n, :, q], flat))
                + decimal(exterior[n, q])
                for q in range(6)
            ]
            for n in range(3)
        ]
        scale = decimal(bank.mesh.scales[cell, triangle])
        coefficients = exact_coefficients(
            jets,
            np.asarray(bank.mesh.local_vertices[cell, triangle]),
            scale,
            np.asarray(bank.mesh.transforms[cell, triangle]),
        )
        return tuple(
            np.asarray(
                [
                    float(v)
                    for v in exact_derivatives(
                        coefficients,
                        point,
                        np.asarray(bank.mesh.origins[cell, triangle]),
                        scale,
                        rank,
                    )
                ]
            ).reshape((2,) * rank)
            for rank in (3, 4)
        )


@pytest.mark.parametrize("state_index", [0, 1], ids=["trial", "converged"])
def test_saddle_derivatives_match_precise_bank(saddle_inputs, state_index):
    bank, cache, roots, states = saddle_inputs
    moments = states[state_index]
    pitch = jnp.median(bank.geometry.pitch)
    solved = fixed_points.solve(
        cache, jnp.stack(moments, axis=-1), roots, roots, pitch, -1.0
    )
    point = solved.position[1]
    assert bool(solved.valid[1])

    def evaluate(bank, moments, point):
        field = fixed_points.field_from_jets(
            support_read.nodal_jets(bank, moments),
            bank.sample_index,
            bank.mesh,
            bank_rows=bank.rows,
            bank_moments=jnp.stack(moments, axis=-1),
            bank_exterior=bank.exterior,
        )
        return topology._curvature_derivatives(field, point, pitch)

    actual = jax.jit(evaluate)(bank, moments, point)
    expected = reference(bank, moments, point)
    for rank, got, want in zip((3, 4), actual, expected):
        error = float(np.max(np.abs(np.asarray(got) - want)))
        print(
            f"DERIVATIVE state={state_index} rank={rank} absolute_error={error:.17g}",
            flush=True,
        )
        np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)
