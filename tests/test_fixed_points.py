"""Stationary-point accuracy, row-window refusal and root-type contracts."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points as fixed
from nova.equilibrium.topology import FieldJet

assert jax.config.jax_enable_x64 is True


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Polynomial:
    """A quartic maximum or cubic axis/saddle field with exact derivatives."""

    cubic: bool = False

    def value(self, point):
        x, y = point - jnp.asarray((0.13, -0.21))
        return jnp.where(
            self.cubic,
            -x * x - y * y + y**3,
            -x * x - y * y - 0.1 * x**4 - 0.2 * x * x * y * y,
        )

    def evaluate(self, point):
        return FieldJet(
            self.value(point),
            jax.grad(self.value)(point),
            jax.hessian(self.value)(point),
        )


def polynomial_cache(cubic=False):
    theta = np.arange(6) * np.pi / 3
    nodes = np.concatenate(
        (np.zeros((1, 2)), np.stack((np.cos(theta), np.sin(theta)), -1))
    )
    mesh = fixed.build_mesh(
        nodes, np.arange(7)[None], SimpleNamespace(pitch=np.ones(1))
    )
    jets = fixed.stack_jet(jax.vmap(Polynomial(cubic).evaluate)(jnp.asarray(nodes)))
    return fixed.RowCache(
        mesh, jnp.asarray(nodes), jnp.arange(7), jnp.zeros((7, 1, 3, 6)), jets
    )


def test_quartic_value_gradient_and_hessian_exactness():
    cache = polynomial_cache()
    field = fixed._CachedField(
        cache.mesh, cache.lookup, cache.rows, jnp.zeros((1, 3)), cache.exterior
    )
    points = jnp.asarray(((0.1, 0.2), (-0.3, -0.1), (0.45, -0.15)))
    actual = jax.jit(jax.vmap(field.evaluate))(points)
    expected = jax.vmap(Polynomial().evaluate)(points)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=2e-13, rtol=0)


def test_analytic_axis_and_saddle_converge_in_small_mesh():
    cache = polynomial_cache(True)
    previous = jnp.asarray(((0.13, -0.21), (0.13, -0.21 + 2 / 3)))
    result = fixed.solve(
        cache, jnp.zeros((1, 3)), previous + 0.035, previous, jnp.asarray(1.0)
    )
    assert np.all(result.valid)
    np.testing.assert_allclose(result.position, previous, atol=1e-10, rtol=0)
    assert np.max(result.iterations) < 12
    assert np.all(result.status == fixed.FixedPointStatus.CONVERGED)


def test_wrong_axis_retries_previous_fixed_point():
    cache = polynomial_cache(True)
    previous = jnp.asarray(((0.13, -0.21), (0.13, -0.21 + 2 / 3)))
    result = fixed.solve(
        cache,
        jnp.zeros((1, 3)),
        previous.at[0].set(previous[1]),
        previous,
        jnp.asarray(1.0),
    )
    assert result.retried[0]
    assert np.all(result.valid)
    np.testing.assert_allclose(result.position, previous, atol=1e-10)


def test_wrong_root_is_refused_when_retry_is_also_a_saddle():
    cache = polynomial_cache(True)
    saddle = jnp.asarray(((0.13, -0.21 + 2 / 3),))
    result = fixed.solve(cache, jnp.zeros((1, 3)), saddle, saddle, jnp.asarray(1.0))
    assert not result.valid[0]
    assert result.status[0] == fixed.FixedPointStatus.WRONG_ROOT
    assert result.retried[0]


def test_missing_rows_return_typed_refusal_without_build(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a solve must never build rows")

    monkeypatch.setattr(fixed, "_row_program", forbidden)
    cache = replace(polynomial_cache(), lookup=jnp.full(7, -1, dtype=jnp.int32))
    seed = jnp.asarray(((0.1, -0.1),))
    result = fixed.solve(cache, jnp.zeros((1, 3)), seed, seed, jnp.asarray(1.0))
    assert not result.valid[0]
    assert result.status[0] == fixed.FixedPointStatus.NEEDS_ROWS


def test_axis_sign_rejects_maximum_when_minimum_is_required():
    cache = polynomial_cache()
    seed = jnp.asarray(((0.13, -0.21),))
    result = fixed.solve(
        cache, jnp.zeros((1, 3)), seed, seed, jnp.asarray(1.0), jnp.asarray(1.0)
    )
    assert not result.valid[0]
    assert result.status[0] == fixed.FixedPointStatus.WRONG_ROOT


@pytest.mark.parametrize("state", [0, 16, 17, 18])
def test_coarse_replay_never_accepts_saddle_as_axis(state):
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/prototypes/fixed_points/coarse-replay.npz"
    )
    with np.load(path) as data:
        mesh = fixed.build_mesh(
            data["nodes"], data["indices"], SimpleNamespace(pitch=data["pitch"])
        )
        jets = jnp.asarray(data[f"jets_{state}"])
        cache = fixed.RowCache(
            mesh,
            jnp.asarray(data["nodes"]),
            jnp.asarray(data[f"lookup_{state}"]),
            jnp.zeros((len(jets), 1, 3, 6)),
            jets,
        )
        seed = jnp.asarray(data[f"seed_{state}"])
        # Replay the accepted wrong root as both candidate and last-known point:
        # a retry cannot authorize the same saddle as an axis.
        wrong = jnp.asarray(data[f"position_{state}"])
        result = fixed.solve(
            cache,
            jnp.zeros((1, 3)),
            seed,
            wrong,
            jnp.asarray(data["pitch"].mean()),
            iterations=24,
        )
    assert not result.valid[0], f"coarse state {state} accepted a saddle as the axis"
    assert result.status[0] == fixed.FixedPointStatus.WRONG_ROOT
    assert result.retried[0]


def test_maintenance_slides_bounded_window_and_reuses_overlap(monkeypatch):
    initial = polynomial_cache()
    nodes = np.asarray(initial.nodes)
    geometry = SimpleNamespace(
        sample_points=np.stack((nodes, nodes + (1.5, 0), nodes + (3, 0))),
        pitch=np.ones(3),
    )
    coupling = SimpleNamespace(centre=np.zeros((1, 2)))
    built = []

    def rows(points, coupling):
        built.extend(np.asarray(points).tolist())
        return jnp.broadcast_to(points[:, 0, None, None, None], (len(points), 1, 3, 6))

    monkeypatch.setattr(fixed, "_row_program", rows)
    cache, first = fixed.maintain_rows(
        geometry, coupling, Polynomial(), jnp.asarray(((0.0, 0.0),)), 0.2, capacity=21
    )
    assert first.nodes_added > 0 and first.nodes_retained == 0
    previous_nodes = np.flatnonzero(np.asarray(cache.lookup) >= 0)
    count = len(built)
    moved, second = fixed.maintain_rows(
        geometry,
        coupling,
        Polynomial(),
        jnp.asarray(((1.5, 0.0),)),
        0.8,
        cache=cache,
        capacity=21,
    )
    assert second.nodes_retained > 0 and second.nodes_added > 0
    assert len(built) - count == 3 * ((second.nodes_added + 2) // 3)
    assert moved.rows.shape == cache.rows.shape == (21, 1, 3, 6)
    assert second.seconds > 0 and second.cache_bytes == moved.rows.nbytes
    for node in previous_nodes:
        slot = int(moved.lookup[node])
        if slot >= 0:
            np.testing.assert_array_equal(
                moved.rows[slot], cache.rows[int(cache.lookup[node])]
            )
    with pytest.raises(ValueError, match="capacity"):
        fixed.maintain_rows(
            geometry,
            coupling,
            Polynomial(),
            jnp.asarray(((1.5, 0.0),)),
            10.0,
            capacity=2,
        )


def test_nonfinite_field_is_not_reported_as_missing_rows():
    cache = replace(polynomial_cache(), exterior=jnp.full((7, 6), jnp.nan))
    seed = jnp.asarray(((0.1, -0.1),))
    result = fixed.solve(cache, jnp.zeros((1, 3)), seed, seed, jnp.asarray(1.0))
    assert not result.valid[0]
    assert result.status[0] != fixed.FixedPointStatus.NEEDS_ROWS


@pytest.mark.parametrize("kind", ["limited", "diverted"])
def test_closed_form_equilibrium_on_small_hex_mesh(kind):
    from tests.equilibrium import test_topology_read as fixture

    _, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, 50)
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    mesh = fixed.build_mesh(nodes, inverse.reshape(samples.shape[:-1]), geometry)
    jets = fixed.stack_jet(jax.vmap(analytic.evaluate)(jnp.asarray(nodes)))
    cache = fixed.RowCache(
        mesh,
        jnp.asarray(nodes),
        jnp.arange(len(nodes)),
        jnp.zeros((len(nodes), 1, 3, 6)),
        jets,
    )
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    pitch = jnp.median(geometry.pitch)
    result = fixed.solve(cache, jnp.zeros((1, 3)), roots + 0.03 * pitch, roots, pitch)
    assert np.all(result.valid)
    assert np.max(np.linalg.norm(np.asarray(result.position - roots), axis=-1)) < float(
        pitch
    )


def test_newton_trajectory_leaves_rows_without_building(monkeypatch):
    first = polynomial_cache(True)
    first_nodes = np.asarray(first.nodes)
    nodes = np.concatenate((first_nodes, first_nodes + (1.5, 0.0)))
    mesh = fixed.build_mesh(
        nodes, np.arange(14).reshape(2, 7), SimpleNamespace(pitch=np.ones(2))
    )
    jets = fixed.stack_jet(
        jax.vmap(Polynomial(True).evaluate)(
            jnp.asarray(first_nodes) - jnp.asarray((1.5, 0.0))
        )
    )
    lookup = jnp.concatenate((jnp.arange(7), jnp.full(7, -1, dtype=jnp.int32)))
    cache = fixed.RowCache(
        mesh, jnp.asarray(nodes), lookup, jnp.zeros((7, 1, 3, 6)), jets
    )

    def forbidden(*args, **kwargs):
        pytest.fail("row build inside Newton")

    monkeypatch.setattr(fixed, "_row_program", forbidden)
    seed = jnp.asarray(((0.2, -0.21),))
    result = fixed.solve(cache, jnp.zeros((1, 3)), seed, seed, jnp.asarray(1.0))
    assert result.iterations[0] == 1
    assert result.position[0, 0] > 0.75
    assert not result.valid[0]
    assert result.status[0] == fixed.FixedPointStatus.NEEDS_ROWS


def _path_carrier():
    """Five cells: axis, control branch, saddle gate, wall contact, spacer."""
    from scripts.prototypes.fast_map_production.test_bottleneck_level import (
        _carrier,
        _square,
    )

    return _carrier(
        [
            _square(1.0, 0.0),  # 0 axis
            _square(0.0, 0.0),  # 1 control branch
            _square(2.0, 0.0),  # 2 saddle gate (the merge-path saddle)
            _square(3.0, 0.0),  # 3 wall contact
            _square(0.0, 1.0),  # 4 spacer
        ]
    )


def _path_roots(geometry, levels):
    """Three roots: axis, the merge-path saddle, and a higher wall-contact root."""
    position = jnp.asarray(
        np.stack(
            (
                np.asarray(geometry.centre[0]),
                np.asarray(geometry.centre[2]),
                np.asarray(geometry.centre[3]),
            )
        )
    )
    jet = SimpleNamespace(value=jnp.asarray(levels), hessian=jnp.zeros((3, 2, 2)))
    return SimpleNamespace(
        position=position, jet=jet, valid=jnp.asarray([True, True, True])
    )


def test_path_admission_rejects_the_wall_contact_root(monkeypatch):
    """The axis component reaches its saddle before a higher private wall root."""
    from nova.equilibrium import contour_tree, topology
    from scripts.prototypes.fast_map_production.kernel import _first_tree_events

    levels = jnp.asarray([1.0, -0.5, 0.3, -0.6])
    tree = contour_tree.ContourTreeArrays(
        node_vertex=jnp.arange(4),
        node_psi=levels,
        node_valid=jnp.ones(4, bool),
        critical_type=jnp.asarray([2, 1, 1, 1]),
        edges=jnp.asarray([[0, 1], [1, 2], [1, 3]]),
        edge_valid=jnp.ones(3, bool),
        overflow=jnp.asarray(False),
    )
    points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    geometry = SimpleNamespace(
        contour_mesh=SimpleNamespace(
            vertex_rz=points,
            vertex_is_wall=jnp.asarray([False, False, True, True]),
            vertex_valid=jnp.ones(4, bool),
            edges=jnp.asarray([[0, 1], [1, 2], [1, 3]]),
            edge_valid=jnp.ones(3, bool),
        ),
        contour_sample_points=points,
        contour_sources=jnp.arange(4)[:, None],
        contour_weights=jnp.ones((4, 1)),
        pitch=jnp.ones(1),
    )
    monkeypatch.setattr(contour_tree, "build_contour_tree", lambda *args: tree)
    monkeypatch.setattr(topology, "_point_values", lambda *args: levels)
    resolved = True

    def polish(*args):
        return SimpleNamespace(
            position=points[1],
            jet=FieldJet(levels[1], jnp.zeros(2), jnp.diag(jnp.asarray([-1.0, 1.0]))),
            valid=jnp.asarray(resolved),
        )

    monkeypatch.setattr(topology, "stationary_read", polish)
    saddle, saddle_live, wall, wall_live, valid = _first_tree_events(
        None, geometry, points[0], jnp.asarray(1.0), None
    )
    assert bool(valid) and bool(saddle_live) and bool(wall_live)
    np.testing.assert_array_equal(saddle.position, points[1])
    np.testing.assert_array_equal(wall, points[3])
    assert int(jnp.argmax(levels[1:3])) == 1  # Level ranking picks the wrong root.

    resolved = False
    _, saddle_live, _, wall_live, valid = _first_tree_events(
        None, geometry, points[0], jnp.asarray(1.0), None
    )
    assert bool(wall_live) and not bool(saddle_live) and not bool(valid)

    wall_first = replace(tree, edges=jnp.asarray([[0, 2], [2, 1], [1, 3]]))
    monkeypatch.setattr(contour_tree, "build_contour_tree", lambda *args: wall_first)
    _, saddle_live, wall, wall_live, valid = _first_tree_events(
        None, geometry, points[0], jnp.asarray(1.0), None
    )
    assert bool(valid) and bool(wall_live) and not bool(saddle_live)
    np.testing.assert_array_equal(wall, points[2])


def test_psi_rounding_enclosure_bounds_the_fit():
    """The enclosure is the binary64 bound of the fitted psi, positive and small."""
    from scripts.prototypes.fast_map_production.kernel import psi_rounding_enclosure

    coefficients = jnp.asarray(np.linspace(-2.0, 3.0, 12).reshape(2, 6))
    bound = float(psi_rounding_enclosure(coefficients))
    assert bound > 0.0
    assert bound < 1e-1
