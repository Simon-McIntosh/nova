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

# The control branch's tangency, declared here and compared against the
# function's output rather than handed in as the expectation itself.
CONTROL_TANGENCY = -0.3


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
        None,
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
    # tangency, so the tangency is kept. The expectation is the declared
    # constant, not the value handed in, and it is asserted to differ from the
    # bottleneck so a maximum composition is rejected here too.
    control = _wall_effective_level(jnp.asarray([CONTROL_TANGENCY]), ceiling[1:2])
    assert float(ceiling[1]) == 0.5
    assert float(control[0]) == CONTROL_TANGENCY
    assert float(control[0]) != float(ceiling[1])


def test_effective_level_is_the_minimum_of_the_two_operands():
    tangency = jnp.asarray([-0.05, 0.2, -1.0, 0.5, 0.7])
    bottleneck = jnp.asarray([-0.5, 0.5, -0.5, 0.5, 0.3])
    effective = np.asarray(_wall_effective_level(tangency, bottleneck))
    expected = np.minimum(np.asarray(tangency), np.asarray(bottleneck))
    # The effective level IS the more scrape-off-side operand, exactly. A
    # minimum of two operands is exact in binary64, so the comparison is
    # bitwise here, not a tolerance test.
    assert np.allclose(effective, expected, atol=0.0, rtol=1e-15)
    assert np.array_equal(effective, expected)


def _max_adjacent_step(values):
    """Largest difference between neighbouring samples of a swept series."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(np.diff(values))))


def _step_at_the_tie(tangency, bottleneck):
    """A deliberately non-continuous variant: a 0.1 step where the operands meet.

    Selection is on ``>=`` at the tie, and the branch not taken is offset, so
    the series jumps at the crossing instead of following the minimum.
    """
    return jnp.where(tangency >= bottleneck, bottleneck, tangency + 0.1)


def test_effective_level_is_continuous_through_the_tie():
    bottleneck = 0.5
    tangency = bottleneck + np.linspace(-1e-6, 1e-6, 41)
    operand_step = _max_adjacent_step(tangency)
    effective = np.asarray(
        _wall_effective_level(
            jnp.asarray(tangency), jnp.full((tangency.size,), bottleneck)
        )
    )
    expected = np.minimum(tangency, bottleneck)
    # Exactness first: the sweep follows the minimum to rtol 1e-15, atol 0.
    assert np.allclose(effective, expected, atol=0.0, rtol=1e-15)
    # Continuity second: the largest adjacent-sample difference cannot exceed
    # the operand step, because the level is one of the two operands and each
    # of those moves by at most the step.
    largest = _max_adjacent_step(effective)
    assert largest <= operand_step, (
        f"level steps by {largest} over an operand step of {operand_step}"
    )

    # The variant carries a step at the tie, so both checks must reject it:
    # this is what shows the checks can fail rather than passing on anything.
    broken = np.asarray(
        _step_at_the_tie(jnp.asarray(tangency), jnp.full((tangency.size,), bottleneck))
    )
    assert not np.allclose(broken, expected, atol=0.0, rtol=1e-15)
    assert _max_adjacent_step(broken) > operand_step


def _selected_wall_candidate(tangency, bottleneck):
    """The read's wall selection over contacts, as the read composes it.

    Per-contact effective level, then the most axis-ward contact wins. This is
    the composition the read performs between its bottleneck field and its wall
    candidate, so an assertion here is an assertion about the read's own
    ``wall_candidate``, not about the level function in isolation.
    """
    level = _wall_effective_level(jnp.asarray(tangency), jnp.asarray(bottleneck))
    return float(level[jnp.argmax(level)]), jnp.asarray(level)


def test_read_wall_candidate_takes_the_bottleneck_limited_level():
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
    tangency = jnp.asarray([levels[3], CONTROL_TANGENCY])
    bottleneck = jnp.asarray([ceiling[3], ceiling[1]])
    assert float(bottleneck[0]) == -0.5
    assert float(bottleneck[1]) == 0.5

    selected, effective = _selected_wall_candidate(tangency, bottleneck)
    # The gated contact carries the saddle level, not its own more axis-ward
    # tangency; the directly reachable control keeps its tangency.
    assert float(effective[0]) == float(bottleneck[0])
    assert float(effective[1]) == CONTROL_TANGENCY
    assert selected == CONTROL_TANGENCY

    # Falsification. Both compositions the read must not use move the selected
    # candidate, so this assertion rejects each: raw tangency hands the contact
    # to the gated cell, and a maximum hands it to the control's bottleneck.
    operand = np.asarray(tangency)
    raw_selected = float(operand[jnp.argmax(operand)])
    raised = np.maximum(operand, np.asarray(bottleneck))
    raised_selected = float(raised[jnp.argmax(raised)])
    assert raw_selected == float(levels[3])
    assert raw_selected != selected
    assert raised_selected == float(bottleneck[1])
    assert raised_selected != selected
