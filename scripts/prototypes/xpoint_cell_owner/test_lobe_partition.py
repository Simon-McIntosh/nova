"""Analytic checks for the lobe-and-remainder ring partition.

The cell fan is built by ``current._normal_rings`` itself, driven with a form
whose separatrix rays are dead, so the builder emits only the straight rows and
the check exercises the same path the booking uses. Every expected value is the
closed form over the polygon's triangles, which shares no quadrature with the
booking's clip rule.
"""

# ruff: noqa: E402
import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

from nova.equilibrium import current
from nova.equilibrium.topology import SaddleNormalForm
from scripts.prototypes.xpoint_cell_owner.lobe_booking import _ring_moments
from scripts.prototypes.xpoint_cell_owner.lobe_reference import triangle_integral

SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
CENTRE = (0.0, 0.0)
DENSITY = (0.5, 0.25, -0.125, 0.375, 0.0625, -0.1875)
STRIPS = 13
SEGMENTS = 4


def _dead_form(centre):
    """A normal form whose rays never reach the cell, so only straight rows live."""
    dead = jnp.full((4, 2), jnp.nan)
    return SaddleNormalForm(
        position=jnp.asarray(centre, dtype=float),
        direction=dead,
        curvature=jnp.zeros((4, 2)),
        cubic=jnp.zeros((4, 2)),
        positive=jnp.zeros(4, dtype=bool),
        valid=jnp.ones(4, dtype=bool),
        reason=jnp.zeros(4),
    )


def _cell_fan(vertices, centre):
    """The straight rows of a one-slot identity-interval ring builder call."""
    vertices = jnp.asarray(vertices, dtype=float)
    count = jnp.asarray(len(vertices))
    intervals = jnp.zeros((2, len(vertices), STRIPS, 2))
    intervals = intervals.at[..., 0, 1].set(1.0)
    selected = jnp.zeros(2, dtype=bool).at[0].set(True)
    rings, sizes = current._normal_rings(
        vertices,
        count,
        jnp.asarray(centre),
        _dead_form(centre),
        selected,
        intervals,
        SEGMENTS,
    )
    straight = 2 * len(vertices) * STRIPS
    return rings[:straight], sizes[:straight]


def _closed_form(vertices, centre, coefficients=DENSITY):
    """Exact area, current and first moments over the star fan of the polygon."""
    vertices = np.asarray(vertices, dtype=float)
    origin = np.asarray(centre, dtype=float)
    total = np.zeros(4)
    for slot in range(len(vertices)):
        triangle = (origin, vertices[slot], vertices[(slot + 1) % len(vertices)])
        total += np.asarray(triangle_integral(triangle, coefficients, origin, 1.0))
    return total


UNIT = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_one_live_slot_carries_the_polygon_once():
    rings, sizes = _cell_fan(SQUARE, CENTRE)
    assert int(np.count_nonzero(np.asarray(sizes))) == len(SQUARE)


def test_cell_fan_matches_the_closed_form_over_the_polygon():
    rings, sizes = _cell_fan(SQUARE, CENTRE)
    origin = jnp.asarray(CENTRE)
    density_at = lambda points: (  # noqa: E731
        DENSITY[0]
        + DENSITY[1] * points[..., 0]
        + DENSITY[2] * points[..., 1]
        + DENSITY[3] * points[..., 0] ** 2
        + DENSITY[4] * points[..., 0] * points[..., 1]
        + DENSITY[5] * points[..., 1] ** 2
    )
    booked = np.asarray(
        _ring_moments(rings, sizes, jnp.ones(rings.shape[0]), origin, density_at)
    )
    dense = _closed_form(SQUARE, CENTRE)
    assert np.all(np.abs(booked - dense) < 1e-13), (booked, dense)


def test_a_negative_slot_subtracts_its_region():
    rings, sizes = _cell_fan(SQUARE, CENTRE)
    inner = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.5, 0.75))
    inner_rings, inner_sizes = _cell_fan(inner, CENTRE)
    origin = jnp.asarray(CENTRE)
    ones = lambda points: jnp.ones(points.shape[:-1])  # noqa: E731
    whole = np.asarray(
        _ring_moments(rings, sizes, jnp.ones(rings.shape[0]), origin, ones)
    )
    part = np.asarray(
        _ring_moments(
            inner_rings, inner_sizes, jnp.ones(inner_rings.shape[0]), origin, ones
        )
    )
    signed = np.asarray(
        _ring_moments(
            jnp.concatenate((rings, inner_rings)),
            jnp.concatenate((sizes, inner_sizes)),
            jnp.concatenate(
                (jnp.ones(rings.shape[0]), -jnp.ones(inner_rings.shape[0]))
            ),
            origin,
            ones,
        )
    )
    assert np.all(np.abs(signed - (whole - part)) < 1e-13)
    expected = _closed_form(SQUARE, CENTRE, UNIT) - _closed_form(inner, CENTRE, UNIT)
    assert np.all(np.abs(signed - expected) < 1e-12)
