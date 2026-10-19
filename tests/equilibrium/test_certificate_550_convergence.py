"""Stationary-root tangents on a small cache and a published carrier mesh."""

import numpy as np
import pytest


def test_stationary_root_tangent_at_exact_seed() -> None:
    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium import fixed_points
    from nova.jax.config import configure_dtypes
    from tests.test_fixed_points import polynomial_cache

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    print(f"ROOT_MODULE={fixed_points.__file__}")
    cache = polynomial_cache()
    rows = cache.rows.at[:, 0, 0, 0].set(0.2 * cache.nodes[:, 1])
    rows = rows.at[:, 0, 0, 2].set(0.2)
    cache = replace(cache, rows=rows)
    initial = jnp.asarray(((0.13, -0.21),))
    zero = jnp.zeros((1, 3))
    seed = fixed_points.solve(cache, zero, initial, initial, jnp.asarray(1.0)).position
    at_seed = fixed_points.solve(cache, zero, seed, seed, jnp.asarray(1.0))
    assert int(at_seed.iterations[0]) == 0

    def root(amplitude):
        moments = jnp.asarray(((amplitude, 0.0, 0.0),))
        result = fixed_points.solve(cache, moments, seed, seed, jnp.asarray(1.0))
        return result.position[0]

    amplitude = jnp.asarray(0.0)
    _, tangent = jax.jvp(root, (amplitude,), (jnp.asarray(1.0),))
    step = 1.0 / 4096.0
    finite = (root(amplitude + step) - root(amplitude - step)) / (2 * step)
    print(f"ROOT_TANGENT jvp={np.asarray(tangent)} central={np.asarray(finite)}")
    np.testing.assert_allclose(tangent, finite, rtol=2e-6, atol=1e-10)
    assert abs(float(tangent[1])) > 0.05


@pytest.mark.slow
def test_certified_limited_carrier_root_tangent() -> None:
    import jax
    import jax.numpy as jnp

    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium import fixed_points, topology
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as fixture
    from scripts.prototypes.support import certificate_fixtures

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    carrier, _, _ = certificate._case("weak-rotation-reactor-static", clip_mode="exact")
    identity = fixture.cache_identity(
        carrier,
        requested_cells=-500,
        wall_nodes=fixture.WALL_POINT_COUNT,
        wall=None,
    )
    digest = certificate_fixtures.fixture_digest(identity)
    assert (
        certificate_fixtures.fixture_entry(certificate_fixtures.CARRIER_KIND, digest)
        is not None
    )
    machine = certificate_fixtures.cached_carrier(carrier, -500, None)
    assert machine.cache["hit"] and len(machine.node) == 553
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    mesh = fixed_points.build_mesh(nodes, inverse.reshape(samples.shape[:-1]), geometry)
    centre = jnp.asarray((6.2, 0.0))

    def quadratic(point):
        delta = point - centre
        return topology.FieldJet(-jnp.dot(delta, delta), -2 * delta, -2 * jnp.eye(2))

    jets = fixed_points.stack_jet(jax.vmap(quadratic)(jnp.asarray(nodes)))
    rows = jnp.zeros((len(nodes), 1, 3, 6), dtype=jnp.float64)
    rows = rows.at[:, 0, 0, 0].set(0.2 * jnp.asarray(nodes[:, 1]))
    rows = rows.at[:, 0, 0, 2].set(0.2)
    cache = fixed_points.RowCache(
        mesh,
        jnp.asarray(nodes),
        jnp.arange(len(nodes), dtype=jnp.int32),
        rows,
        jets,
    )
    pitch = jnp.median(geometry.pitch)
    initial = centre[None]
    zero = jnp.zeros((1, 3))
    seed = fixed_points.solve(cache, zero, initial, initial, pitch).position
    at_seed = fixed_points.solve(cache, zero, seed, seed, pitch)
    assert bool(at_seed.valid[0]) and int(at_seed.iterations[0]) == 0

    def root(amplitude):
        moments = jnp.asarray(((amplitude, 0.0, 0.0),))
        return fixed_points.solve(cache, moments, seed, seed, pitch).position[0]

    amplitude = jnp.asarray(0.0)
    _, tangent = jax.jvp(root, (amplitude,), (jnp.asarray(1.0),))
    step = 1 / 4096
    finite = (root(amplitude + step) - root(amplitude - step)) / (2 * step)
    print(
        f"CARRIER_ROOT_TANGENT jvp={np.asarray(tangent)} central={np.asarray(finite)}"
    )
    np.testing.assert_allclose(tangent, finite, rtol=2e-6, atol=1e-10)
    assert abs(float(tangent[1])) > 0.05
