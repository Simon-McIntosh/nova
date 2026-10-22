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


def test_composite_support_is_continuous_across_disk_entry_and_exit():
    assert jax.config.jax_enable_x64 is True
    radii = (0.0, 0.25, 0.5, 0.9, 1.1, 1.5, 3.0)
    fractions = [_disk_weight_fraction(jnp.asarray(r)) for r in radii]
    # The intersection measure proves the disk actually entered and left.
    assert fractions[0] == pytest.approx(0.0, abs=1e-12)
    assert fractions[1] > 0.0
    assert fractions[-1] == pytest.approx(1.0, abs=1e-12)

    moments = [
        _moments(jnp.asarray(r), force_conic_right=FORCE_CONIC_RIGHT) for r in radii
    ]
    reference = moments[0]
    for moment in moments:
        assert jnp.all(jnp.isfinite(moment))
        assert jnp.allclose(moment, reference, rtol=0.0, atol=1e-12)


def test_force_conic_right_breaks_the_continuity_assertion():
    """The declared negative control: the whole ordinary cut on the right fails."""
    assert jax.config.jax_enable_x64 is True
    radii = (0.0, 0.25, 0.5, 0.9, 1.1, 1.5, 3.0)
    moments = [_moments(jnp.asarray(r), force_conic_right=True) for r in radii]
    reference = moments[0]
    moved = [float(jnp.max(jnp.abs(moment - reference))) for moment in moments]
    # The mutated booking is not continuous: its current grows with the disk.
    assert max(moved) > 1e-3, moved