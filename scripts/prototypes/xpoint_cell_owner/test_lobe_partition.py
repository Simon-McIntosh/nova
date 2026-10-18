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
import pytest
from pathlib import Path
from types import SimpleNamespace

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

from nova.equilibrium import current
from nova.equilibrium.topology import SaddleNormalForm
from scripts.prototypes.xpoint_cell_owner.lobe_booking import (
    _ring_moments,
    _whole_cell_rings,
    remaining_moments,
)
from scripts.prototypes.xpoint_cell_owner.lobe_reference import (
    lobe_reference,
    triangle_integral,
)
from scripts.prototypes.xpoint_cell_owner import lobe_reference_probe as probe

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


def _density_at(points):
    x, y = points[..., 0], points[..., 1]
    return (
        DENSITY[0]
        + DENSITY[1] * x
        + DENSITY[2] * y
        + DENSITY[3] * x**2
        + DENSITY[4] * x * y
        + DENSITY[5] * y**2
    )


UNIT = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_one_live_slot_carries_the_polygon_once():
    rings, sizes = _cell_fan(SQUARE, CENTRE)
    assert int(np.count_nonzero(np.asarray(sizes))) == len(SQUARE)


def test_cell_fan_matches_the_closed_form_over_the_polygon():
    rings, sizes = _cell_fan(SQUARE, CENTRE)
    origin = jnp.asarray(CENTRE)
    booked = np.asarray(
        _ring_moments(rings, sizes, jnp.ones(rings.shape[0]), origin, _density_at)
    )
    dense = _closed_form(SQUARE, CENTRE)
    assert np.all(np.abs(booked - dense) < 1e-13), (booked, dense)


def test_a_negative_slot_subtracts_its_region():
    rings, sizes = _cell_fan(SQUARE, CENTRE)
    inner = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.5, 0.75))
    inner_rings, inner_sizes = _cell_fan(inner, CENTRE)
    origin = jnp.asarray(CENTRE)
    signed = np.asarray(
        _ring_moments(
            jnp.concatenate((rings, inner_rings)),
            jnp.concatenate((sizes, inner_sizes)),
            jnp.concatenate(
                (jnp.ones(rings.shape[0]), -jnp.ones(inner_rings.shape[0]))
            ),
            origin,
            _density_at,
        )
    )
    whole = _closed_form(SQUARE, CENTRE)
    part = _closed_form(inner, CENTRE)
    np.testing.assert_allclose(signed, whole - part, rtol=0, atol=1e-12)


def test_remainder_uses_the_ring_builders_counted_polygon():
    """A padded fourth vertex must never replace the triangle's closing edge."""
    vertices = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (9.0, 9.0))
    geometry = SimpleNamespace(
        vertices=jnp.asarray([vertices]),
        vertex_count=jnp.asarray([3]),
        centre=jnp.asarray([[0.0, 0.0]]),
        pitch=jnp.asarray([1.0]),
    )
    reading = SimpleNamespace(
        saddle_form=_dead_form(CENTRE),
        fragment_selected=jnp.zeros((1, 2), dtype=bool),
        edge_interval=jnp.zeros((1, 2, 4, STRIPS, 2)),
    )
    rings, sizes = _whole_cell_rings(geometry, reading, 0, SEGMENTS)
    ones = lambda points: jnp.ones(points.shape[:-1])  # noqa: E731
    whole = np.asarray(
        _ring_moments(rings, sizes, jnp.ones(len(sizes)), geometry.centre[0], ones)
    )
    remainder, valid = remaining_moments(
        geometry,
        reading,
        None,
        jnp.asarray([0]),
        jnp.asarray([UNIT]),
        1.0,
        1.0,
        segments=SEGMENTS,
    )
    expected = _closed_form(vertices[:3], CENTRE, UNIT)
    assert bool(valid[0])
    np.testing.assert_allclose(whole, expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(remainder[0]), expected, rtol=0, atol=1e-13)


def test_lobe_reference_constructs_its_region_without_booking_builder(monkeypatch):
    vertices = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (9.0, 9.0))
    geometry = SimpleNamespace(
        vertices=jnp.asarray([vertices]),
        vertex_count=jnp.asarray([3]),
        centre=jnp.asarray([[0.0, 0.0]]),
        pitch=jnp.asarray([1.0]),
    )
    intervals = jnp.zeros((1, 2, 4, STRIPS, 2))
    intervals = intervals.at[0, 0, :3, 0, 1].set(1.0)
    reading = SimpleNamespace(
        saddle_form=_dead_form(CENTRE),
        fragment_selected=jnp.asarray([[True, False]]),
        edge_interval=intervals,
    )

    def forbidden_builder(*args, **kwargs):
        raise AssertionError("the independent reference called the booking builder")

    monkeypatch.setattr(current, "_normal_rings", forbidden_builder)
    measured, triangles, count = lobe_reference(
        geometry, reading, 0, jnp.asarray([UNIT]), segments=SEGMENTS
    )
    assert count == len(triangles) == 3
    np.testing.assert_allclose(
        measured, _closed_form(vertices[:3], CENTRE, UNIT), rtol=0, atol=1e-13
    )


@pytest.mark.parametrize(
    ("owner_present", "valid", "offset"),
    ((False, True, 0.0), (True, False, 0.0), (True, True, 2e-12)),
)
def test_reference_probe_refuses_missing_invalid_or_inaccurate_rows(
    monkeypatch, owner_present, valid, offset
):
    selected = np.zeros(282, dtype=bool)
    selected[0] = owner_present
    reading = SimpleNamespace(
        booking_cells=selected,
        membership=np.ones(282),
        boundary_flux=1.0,
        axis_flux=0.0,
    )
    built = SimpleNamespace(
        reference=np.zeros((282, 3)),
        reading=lambda _: reading,
        geometry=object(),
        profile=None,
        convention=SimpleNamespace(sigma=1.0),
    )
    monkeypatch.setattr(probe, "build", lambda *_: built)
    monkeypatch.setattr(probe.program, "_density_coefficients", lambda *_: None)
    monkeypatch.setattr(
        probe,
        "lobe_moments",
        lambda *_, **__: (jnp.asarray([[offset, 0.0, 0.0, 0.0]]), jnp.asarray([valid])),
    )
    monkeypatch.setattr(
        probe,
        "remaining_moments",
        lambda *_, **__: (jnp.zeros((1, 4)), jnp.asarray([True])),
    )
    monkeypatch.setattr(probe, "lobe_reference", lambda *_, **__: (np.zeros(4), (), 0))
    monkeypatch.setattr(probe, "polygon_reference", lambda *_, **__: np.zeros(4))
    with pytest.raises(AssertionError):
        probe.run_row(250, Path("unused-cache"), SEGMENTS)
