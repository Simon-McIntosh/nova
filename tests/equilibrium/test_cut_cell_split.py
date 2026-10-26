"""Which region a cut cell books across the normal-form neighbourhood boundary.

A cut cell books one binary support, ``(C ∩ D ∩ N) ∪ (C ∩ Dᶜ ∩ Q)``: the
selected normal-form portion inside the effective disk and the ordinary
quadratic cut outside it, with the same density at the same clip-quadrature
nodes. This module moves the cell across the disk-entry and disk-exit thresholds
and requires the booked current and both first moments to stay continuous.

The literal negative control is the ``--force-conic-right`` mutation: the
right-hand (exterior) support replaced by the whole ordinary cut at the same
density and weight. That mutation must fail the same continuity assertion.
"""

from dataclasses import dataclass

from nova.jax.config import configure_dtypes

configure_dtypes()

import os  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from nova.equilibrium import current  # noqa: E402

# The literal mutation the continuity check must refuse. Supplying it in the
# environment applies ``--force-conic-right``: the right-hand (exterior) support
# replaced by the whole ordinary cut at the same density and weight.
FORCE_CONIC_RIGHT = os.environ.get("NOVA_FORCE_CONIC_RIGHT") == "--force-conic-right"

# A unit square cell, given as its boundary polygon. The clip-quadrature owner
# fans it from its first vertex, so the four boundary points are the ring.
SQUARE = jnp.asarray(
    ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)), dtype=jnp.float64
)


@dataclass(frozen=True)
class LinearDensity:
    """A profile that makes both first moments non-trivial."""

    def current_density(self, radius, normalized):
        return radius + 2.0


@dataclass(frozen=True)
class UnitDensity:
    """Unit density, so the booked current is the support's own area."""

    def current_density(self, radius, normalized):
        return jnp.ones_like(radius)


def _ring(values):
    return values[None], jnp.asarray([values.shape[0]], dtype=jnp.int32)


def _moments(radius, *, force_conic_right=False):
    """Book the square cell split by a disk of the given radius about the origin.

    Both supports are the square itself, so the composite support is the whole
    square for any radius and its moments are the square's moments. The mutation
    replaces the exterior support by the whole square, so the composite grows
    with the disk and its moments jump.
    """
    rings, counts = _ring(SQUARE)
    centre = jnp.zeros(2, dtype=jnp.float64)
    coefficient = jnp.zeros(6, dtype=jnp.float64)
    return current._disk_split_moments(
        rings,
        counts,
        centre,
        jnp.asarray(1.0, dtype=jnp.float64),
        coefficient,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        LinearDensity(),
        centre,
        jnp.asarray(radius, dtype=jnp.float64),
        keep_inside=True,
        force_whole=False,
    ) + current._disk_split_moments(
        rings,
        counts,
        centre,
        jnp.asarray(1.0, dtype=jnp.float64),
        coefficient,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        LinearDensity(),
        centre,
        jnp.asarray(radius, dtype=jnp.float64),
        keep_inside=False,
        force_whole=force_conic_right,
    )


def _disk_weight_fraction(radius):
    """Fraction of the shared quadrature weight inside the disk.

    This is the intersection measure that proves the disk actually entered or
    left the cell across the swept radius.
    """
    rings, counts = _ring(SQUARE)
    centre = jnp.zeros(2, dtype=jnp.float64)
    from nova.equilibrium.clip_quadrature import _quadrature_from_arrays

    points, weights = _quadrature_from_arrays(
        rings, counts, jnp.broadcast_to(centre, (1, 2)), counts >= 3, signed=True
    )
    inside = jnp.sum((points - centre) ** 2, axis=-1) <= radius**2
    total = jnp.sum(weights)
    return float(jnp.sum(jnp.where(inside, weights, 0.0)) / total)


# The cell and the disk the inside piece is measured against. The cell is a
# triangle that is not symmetric about the disk, so the booked first moments are
# non-zero and the density, which reads the absolute x coordinate, is not
# constant across the region.
CELL = jnp.asarray(((-0.8, -0.9), (0.9, -0.9), (0.0, 0.9)), dtype=jnp.float64)
DISK_CENTRE = jnp.asarray((0.1, -0.2), dtype=jnp.float64)
CELL_CENTRE = jnp.zeros(2, dtype=jnp.float64)

# The clip quadrature's own discretisation against this reference, measured at
# the radii below: 8e-6 relative at radius 0.2 and 1.4e-5 at 0.9, i.e. about
# 4e-5 absolute on the largest booked value. The tolerance is set above that
# and far below the sliver the negative control drops.
TOLERANCE = 2e-4


def _inside(radius, *, scale=1.0, profile=None):
    """Book only the inside piece: the cell clipped to the disk.

    The exterior piece is not added, so this is the quantity the split has to
    get right and not a sum in which the inside piece cancels.
    """
    rings, counts = _ring(CELL)
    return current._disk_split_moments(
        rings,
        counts,
        CELL_CENTRE,
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.zeros(6, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        LinearDensity() if profile is None else profile,
        DISK_CENTRE,
        jnp.asarray(radius * scale, dtype=jnp.float64),
        keep_inside=True,
    )


def _polar_reference(radius, *, scale=1.0, unit=False):
    """The inside piece in closed polar form, sharing nothing with the split.

    The region is the convex cell intersected with the disk boundary the clip
    draws, an inscribed regular polygon of ``DISK_ARC_SEGMENTS`` vertices at
    ``radius*scale``. About the disk centre, a ray at angle ``theta`` leaves the
    region at the nearer of the cell edge and that polygon, whose boundary
    radius is ``R cos(pi/n) / cos(phi)`` with ``phi`` the angular distance to the
    nearest polygon vertex. The cell edge is fixed between consecutive vertex
    angles, so each angular piece is smooth and is integrated with
    Gauss-Legendre nodes. The booked
    quantities are ``integral density dA`` and the two ``integral density *
    (x - centre) dA`` with ``density = x + 2``, which in these coordinates is
    ``A + t cos(theta)`` with ``A = disk_centre_x + 2``. With ``unit`` the
    density is one, so the same routine returns the support's area and its
    geometric first moments.
    """
    import numpy as np

    limit = radius * scale
    vertices = np.asarray(CELL, dtype=float)
    origin = np.asarray(DISK_CENTRE, dtype=float)
    centre = np.asarray(CELL_CENTRE, dtype=float)
    density_offset = origin[0] + 2.0
    shift = origin - centre

    angles = np.arctan2(vertices[:, 1] - origin[1], vertices[:, 0] - origin[0])
    order = np.argsort(angles)
    angles = angles[order]
    vertices = vertices[order]
    nodes, weights = np.polynomial.legendre.leggauss(48)

    total = np.zeros(3)
    for index in range(len(angles)):
        start = angles[index]
        stop = angles[(index + 1) % len(angles)]
        if stop <= start:
            stop += 2.0 * np.pi
        first = vertices[index]
        edge = vertices[(index + 1) % len(vertices)] - first
        theta = 0.5 * (stop - start) * nodes + 0.5 * (stop + start)
        jacobian = 0.5 * (stop - start)
        direction = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
        denominator = direction[..., 0] * edge[1] - direction[..., 1] * edge[0]
        numerator = (first[0] - origin[0]) * edge[1] - (first[1] - origin[1]) * edge[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = np.where(np.abs(denominator) > 0.0, numerator / denominator, np.inf)
        segments = current.DISK_ARC_SEGMENTS
        step = 2.0 * np.pi / segments
        phase = np.mod(theta, step) - 0.5 * step
        boundary = limit * np.cos(0.5 * step) / np.cos(phase)
        rho = np.minimum(boundary, np.maximum(hit, 0.0))
        cosine, sine = np.cos(theta), np.sin(theta)
        if unit:
            piece = 0.5 * rho**2
            moment_x = rho**3 * cosine / 3.0 + shift[0] * rho**2 / 2.0
            moment_y = rho**3 * sine / 3.0 + shift[1] * rho**2 / 2.0
        else:
            piece = 0.5 * density_offset * rho**2 + cosine * rho**3 / 3.0
            moment_x = (
                density_offset * rho**3 * cosine / 3.0
                + density_offset * shift[0] * rho**2 / 2.0
                + rho**4 * cosine**2 / 4.0
                + shift[0] * rho**3 * cosine / 3.0
            )
            moment_y = (
                density_offset * rho**3 * sine / 3.0
                + density_offset * shift[1] * rho**2 / 2.0
                + rho**4 * sine * cosine / 4.0
                + shift[1] * rho**3 * cosine / 3.0
            )
        total += jacobian * np.asarray(
            (
                np.sum(weights * piece),
                np.sum(weights * moment_x),
                np.sum(weights * moment_y),
            )
        )
    return jnp.asarray(total, dtype=jnp.float64)


@pytest.mark.parametrize("radius", [0.2, 0.55, 0.9, 1.4, 3.0])
def test_inside_piece_books_the_exact_clipped_disk(radius):
    """The inside piece is asserted against a reference that shares no split.

    The composite support tiles the cell for every radius, so asserting the
    composite against itself cannot fail. This asserts the inside piece -- the
    cell clipped to the disk -- against its closed polar form, at radii that put
    the disk wholly inside the cell, crossing one edge, crossing two, and
    containing the cell.
    """
    assert jax.config.jax_enable_x64 is True
    booked = _inside(radius)
    reference = _polar_reference(radius)
    assert jnp.all(jnp.isfinite(booked))
    assert jnp.allclose(booked, reference, rtol=0.0, atol=TOLERANCE), (
        booked,
        reference,
    )

    # The negative control: where the disk boundary actually crosses the cell, a
    # split that drops a sliver at the crossing must fail this same assertion,
    # so the tolerance is not hiding the region. Where the disk contains the
    # cell there is no crossing to drop a sliver at, so the control does not
    # apply and the check is that the whole cell books exactly.
    fraction = _disk_weight_fraction(radius)
    if 0.0 < fraction < 1.0:
        sliver = _inside(radius, scale=1.0 - 1e-3)
        assert not jnp.allclose(sliver, reference, rtol=0.0, atol=TOLERANCE), radius
    else:
        assert fraction == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("radius", [0.2, 0.55, 0.9, 1.4])
def test_area_current_and_moments_share_one_support(radius):
    """The read area, the booked current and both first moments cover one region.

    The same rings are booked twice, once with unit density and once with
    ``density = x + 2``. Under unit density slot zero is the support's area and
    the two later slots are its geometric first moments about the cell centre,
    so the density-weighted current is fixed by them:

        integral (x + 2) dA = integral (x - centre_x) dA + (centre_x + 2) * area.

    If the two bookings integrated different supports that identity would not
    hold, and both bookings are separately asserted against the closed polar
    form, so neither is merely self-consistent.
    """
    assert jax.config.jax_enable_x64 is True
    support = _inside(radius, profile=UnitDensity())
    weighted = _inside(radius)
    unit_reference = _polar_reference(radius, unit=True)
    assert jnp.all(jnp.isfinite(support))
    assert jnp.allclose(support, unit_reference, rtol=0.0, atol=TOLERANCE), (
        support,
        unit_reference,
    )

    area, moment_x, _ = (float(value) for value in support)
    assert area > 0.0
    predicted = moment_x + (float(CELL_CENTRE[0]) + 2.0) * area
    assert float(weighted[0]) == pytest.approx(predicted, rel=0.0, abs=2.0 * TOLERANCE)


def test_inside_piece_is_continuous_across_disk_entry_and_exit():
    assert jax.config.jax_enable_x64 is True
    radii = (0.0, 0.25, 0.5, 0.9, 1.1, 1.5, 3.0)
    fractions = [_disk_weight_fraction(jnp.asarray(r)) for r in radii]
    # The intersection measure proves the disk actually entered and left.
    assert fractions[0] == pytest.approx(0.0, abs=1e-12)
    assert fractions[1] > 0.0
    assert fractions[-1] == pytest.approx(1.0, abs=1e-12)

    for radius in radii:
        booked = _inside(radius)
        reference = _polar_reference(radius)
        assert jnp.all(jnp.isfinite(booked))
        assert jnp.allclose(booked, reference, rtol=0.0, atol=TOLERANCE), (
            booked,
            reference,
        )


def test_force_conic_right_breaks_the_continuity_assertion():
    """The declared negative control: the whole ordinary cut on the right fails."""
    assert jax.config.jax_enable_x64 is True
    radii = (0.0, 0.25, 0.5, 0.9, 1.1, 1.5, 3.0)
    moments = [_moments(jnp.asarray(r), force_conic_right=True) for r in radii]
    reference = moments[0]
    moved = [float(jnp.max(jnp.abs(moment - reference))) for moment in moments]
    # The mutated booking is not continuous: its current grows with the disk.
    assert max(moved) > 1e-3, moved


# The value test: the region the clip books, against shapely.

# A convex ring that is not a square, and a non-convex one whose disk-crossing
# cut removes the notch corner.
TRIANGLE = jnp.asarray(((-0.8, -0.9), (0.9, -0.9), (0.0, 0.9)), dtype=jnp.float64)
NOTCH = jnp.asarray(
    (
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, -0.2),
        (-0.2, -0.2),
        (-0.2, 1.0),
        (-1.0, 1.0),
    ),
    dtype=jnp.float64,
)


RING_AROUND_DISK = jnp.asarray(
    ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)), dtype=jnp.float64
)


def _signed_region(points):
    """Signed area and first moments of one closed polygon, exact in its vertices."""
    following = jnp.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    area = 0.5 * jnp.sum(cross)
    moment = (
        jnp.stack(
            (
                jnp.sum(cross * (points[:, 0] + following[:, 0])),
                jnp.sum(cross * (points[:, 1] + following[:, 1])),
            )
        )
        / 6.0
    )
    return area, moment


def _piece_moments(vertices, live):
    """Signed area and first moments of every clipped piece, summed.

    Every piece is closed on its own live prefix, so a padded slot and a dead
    piece contribute exactly zero.
    """
    vertices = vertices.reshape(-1, vertices.shape[-2], 2)
    live = live.reshape(-1)
    width = vertices.shape[1]
    slot = jnp.arange(width)[None, :]
    alive = slot < live[:, None]
    following = jnp.where(slot + 1 >= live[:, None], 0, slot + 1)
    ahead = jnp.take_along_axis(vertices, following[..., None], axis=1)
    cross = vertices[..., 0] * ahead[..., 1] - ahead[..., 0] * vertices[..., 1]
    cross = jnp.where(alive & (following < live[:, None]), cross, 0.0)
    area = 0.5 * jnp.sum(cross)
    moment = (
        jnp.stack(
            (
                jnp.sum(cross * (vertices[..., 0] + ahead[..., 0])),
                jnp.sum(cross * (vertices[..., 1] + ahead[..., 1])),
            )
        )
        / 6.0
    )
    return area, moment


def _reference_moments(ring, centre, radius):
    """Shapely's ring ∩ inscribed polygon, in the ring's own orientation."""
    import numpy as np
    from shapely.geometry import Polygon

    angle = (
        2.0 * jnp.pi * jnp.arange(current.DISK_ARC_SEGMENTS) / current.DISK_ARC_SEGMENTS
    )
    clip = np.stack(
        (
            centre[0] + radius * np.cos(angle),
            centre[1] + radius * np.sin(angle),
        ),
        axis=-1,
    )
    subject = Polygon(np.asarray(ring, dtype=float))
    intersection = subject.intersection(Polygon(clip))
    area = float(intersection.area)
    orientation = float(_signed_region(jnp.asarray(ring))[0])
    if area == 0.0:
        return jnp.zeros(3, dtype=jnp.float64)
    centroid = intersection.centroid
    scaled = area * (1.0 if orientation > 0.0 else -1.0)
    return jnp.asarray(
        (
            scaled,
            scaled * centroid.x,
            scaled * centroid.y,
        ),
        dtype=jnp.float64,
    )


def _clipped(ring, count, centre, radius, capacity):
    vertices = jnp.zeros((1, capacity, 2), dtype=jnp.float64)
    vertices = vertices.at[0, :count].set(ring)
    clipped, live = current._disk_clipped_rings(
        vertices,
        jnp.asarray([count], dtype=jnp.int32),
        jnp.asarray(centre, dtype=jnp.float64),
        jnp.asarray(radius, dtype=jnp.float64),
    )
    return _piece_moments(clipped, live)


@pytest.mark.parametrize(
    "ring, centre, radius",
    [
        (SQUARE, (0.0, 0.0), 0.6),
        (TRIANGLE, (0.1, -0.2), 0.7),
        (NOTCH, (0.2, 0.35), 0.7),
        (NOTCH, (-0.6, 0.6), 0.5),
        (NOTCH, (4.0, 4.0), 0.5),
        (SQUARE, (0.0, 0.0), 3.0),
        (RING_AROUND_DISK, (0.0, 0.0), 0.6),
    ],
)
def test_clipped_pieces_book_the_disk_intersection(ring, centre, radius):
    """The clipped pieces' summed region is the ring ∩ the inscribed polygon.

    The disk is placed to cross a convex ring, to cross the non-convex ring
    through its notch, to sit wholly inside the non-convex ring, and to miss the
    ring entirely, where the clipped region must be exactly zero. A ring whose
    every vertex is inside the disk books whole, and a ring that surrounds the
    disk has every vertex outside it while still intersecting it, which a
    vertex-only inside test reads the wrong way round. The ring is padded beyond
    its vertex count, so the pad slots must book nothing.
    """
    assert jax.config.jax_enable_x64 is True
    count = int(ring.shape[0])
    capacity = count + 3
    area, moment = _clipped(ring, count, centre, radius, capacity)
    reference = _reference_moments(ring, centre, radius)
    booked = jnp.stack((area, *moment))
    assert jnp.all(jnp.isfinite(booked)), booked
    assert jnp.allclose(booked, reference, rtol=0.0, atol=1e-9), (
        booked,
        reference,
    )


def test_a_padded_dead_ring_books_nothing():
    """A ring whose count is below three contributes no clipped region."""
    assert jax.config.jax_enable_x64 is True
    capacity = NOTCH.shape[0] + 3
    vertices = jnp.zeros((2, capacity, 2), dtype=jnp.float64)
    vertices = vertices.at[0, : NOTCH.shape[0]].set(NOTCH)
    vertices = vertices.at[1, 0].set(jnp.asarray((0.2, 0.1)))
    vertices = vertices.at[1, 1].set(jnp.asarray((0.3, 0.1)))
    centre = jnp.asarray((0.2, 0.35), dtype=jnp.float64)
    radius = jnp.asarray(0.7, dtype=jnp.float64)
    clipped, live = current._disk_clipped_rings(
        vertices, jnp.asarray([NOTCH.shape[0], 2], dtype=jnp.int32), centre, radius
    )
    area, _ = _piece_moments(clipped, live)
    single, _ = _clipped(NOTCH, NOTCH.shape[0], (0.2, 0.35), 0.7, capacity)
    assert float(area) == pytest.approx(float(single), abs=0.0)
