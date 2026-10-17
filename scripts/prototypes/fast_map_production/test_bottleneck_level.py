"""The wall contact's effective level is the more scrape-off-side of its own
tangency level and the axis-path bottleneck.

A private-flux contact can carry a tangency level more axis-ward than the
X-point while being reachable from the axis only through the saddle. It must
then take the saddle's level, not its own. A directly reachable contact keeps
its tangency level. The carriers below are hand-built quad meshes so the test
is small, CPU-only and independent of the analytic certificate machinery.
"""

# ruff: noqa: E402
from typing import NamedTuple

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.fast_map_production.kernel import (
    _wall_effective_level,
    _wall_path_ceiling,
)

assert jax.config.jax_enable_x64 is True


class Carrier(NamedTuple):
    """Minimal cell mesh: vertices, edge neighbours and cell geometry."""

    vertices: jax.Array
    vertex_count: jax.Array
    centre: jax.Array
    pitch: jax.Array
    neighbour: jax.Array
    neighbour_edge: jax.Array


def _carrier(cells):
    """Assemble a quad carrier from authored cell polygons.

    Cells are indexed in the order given. Edge neighbours are recovered from
    shared vertices, so an isolated cell has neighbour ``-1`` everywhere.
    """
    order = np.array([0, 1, 2, 3])
    vertices = np.stack(cells)
    centre = vertices.mean(axis=1)
    edge_pairs = {}
    neighbour = np.full((len(cells), 4), -1, dtype=np.int32)
    neighbour_edge = np.zeros((len(cells), 4), dtype=np.int32)
    for index, polygon in enumerate(vertices):
        for edge in order:
            pair = (edge, (edge + 1) % 4)
            key = tuple(sorted((tuple(polygon[pair[0]]), tuple(polygon[pair[1]]))))
            if key in edge_pairs:
                peer, peer_edge = edge_pairs.pop(key)
                neighbour[index, edge] = peer
                neighbour_edge[index, edge] = peer_edge
                neighbour[peer, peer_edge] = index
                neighbour_edge[peer, peer_edge] = edge
            else:
                edge_pairs[key] = (index, edge)
    pitch = np.full(len(cells), np.sqrt(np.median(np.abs(_signed_areas(vertices)))))
    return Carrier(
        jnp.asarray(vertices),
        jnp.asarray(np.full(len(cells), 4, dtype=np.int32)),
        jnp.asarray(centre),
        jnp.asarray(pitch),
        jnp.asarray(neighbour),
        jnp.asarray(neighbour_edge),
    )


def _signed_areas(vertices):
    x, y = vertices[..., 0], vertices[..., 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    return 0.5 * np.sum(x * y_next - y * x_next, axis=1)


def _square(x, y):
    return np.array([[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]])


def _constant_levels(values):
    """Cell coefficients of a spatially constant quadratic flux per cell."""
    coefficients = np.zeros((len(values), 6))
    coefficients[:, 0] = values
    return jnp.asarray(coefficients)


def _bottleneck(geometry, levels):
    """Maximin of the edge passage peaks from the axis seed at cell 0."""
    return _wall_path_ceiling(
        geometry,
        _constant_levels(levels),
        None,
        jnp.asarray(geometry.centre[0]),
        0.0,
        False,
        -jnp.inf,
        0,
        1.0,
        False,
    )


def test_private_flux_contact_takes_the_saddle_level():
    # Cell 0 axis, cell 1 control branch, cell 2 the saddle gate, cell 3 the
    # private-flux contact reachable only through cell 2. A spacer cell breaks
    # the collinearity the production mesh builder rejects.
    geometry = _carrier(
        [
            _square(1.0, 0.0),  # 0 axis
            _square(0.0, 0.0),  # 1 control branch
            _square(2.0, 0.0),  # 2 saddle gate
            _square(3.0, 0.0),  # 3 private-flux contact
            _square(0.0, 1.0),  # 4 spacer
        ]
    )
    levels = np.array([1.0, 0.5, -0.5, -0.05, -2.0])
    ceiling = _bottleneck(geometry, levels)
    assert float(ceiling[3]) == -0.5, f"private bottleneck {float(ceiling[3])}"

    saddle_level = levels[2]
    private_tangency = levels[3]
    assert private_tangency > saddle_level  # tangency more axis-ward than saddle
    effective = _wall_effective_level(jnp.asarray([private_tangency]), ceiling[3:4])
    assert float(effective[0]) == saddle_level
    # The tangency alone would have won by 0.45; the bottleneck removes it.
    assert float(effective[0]) < private_tangency

    # Control: cell 1 is directly reachable, its bottleneck sits above its
    # tangency, so the tangency is kept.
    control_tangency = -0.3
    control = _wall_effective_level(jnp.asarray([control_tangency]), ceiling[1:2])
    assert float(ceiling[1]) == 0.5
    assert float(control[0]) == control_tangency


def test_effective_level_is_continuous_in_both_operands():
    tangency = jnp.asarray([-0.05, 0.2, -1.0])
    bottleneck = jnp.asarray([-0.5, 0.5, -0.5])
    effective = _wall_effective_level(tangency, bottleneck)
    assert np.allclose(np.asarray(effective), [-0.5, 0.2, -1.0])
    # Varying the tangency through the bottleneck crossing moves the level
    # continuously: no step.
    swept = _wall_effective_level(
        0.5 + jnp.linspace(-1e-6, 1e-6, 5), jnp.full((5,), 0.5)
    )
    assert np.allclose(np.asarray(swept), 0.5 + np.linspace(-1e-6, 1e-6, 5))
