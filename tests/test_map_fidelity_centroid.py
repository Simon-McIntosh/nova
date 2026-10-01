"""Round-trip contract for the map-fidelity support-current centroid.

The map-fidelity benchmark recovers physical first moments about each cell
centroid through the inverse of the coupling transform before reporting a
support-current centroid.  This contract checks that recovery on a distribution
whose centroid is known in closed form: a current density linear in R across a
rectangle, for which the cell sum of (current at the cell centroid plus the first
moment about it) equals the exact integral centroid.  Because the net first
moment is nonzero, a perturbation of the inverse moves the centroid and must be
detected.  The perturbation is environment-driven so a red arm can be captured as
a negative control without editing the source.
"""

from __future__ import annotations

import os

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

from benchmarks.plasma_cell_map_fidelity import (  # noqa: E402
    current_centroid,
    recover_physical_first_moments,
)

R0, R1, Z0, Z1 = 5.0, 7.0, -1.5, 1.0
ALPHA, BETA = 2.0e6, 3.0e5  # J(R) = ALPHA + BETA * (R - R0)  [A/m^2]
ND, NZ = 7, 5
TOLERANCE = 1.0e-12  # metres
PERTURBATION = 1.0e-2


def _rectangle_distribution():
    """Cell currents, first moments about cell centroids, and the exact centroid.

    J is independent of Z, so every cell's vertical first moment about its own
    centroid is zero and the exact centroid is (integral of R*J / integral of J,
    mid-height).
    """
    radial_edges = np.linspace(R0, R1, ND + 1)
    vertical_edges = np.linspace(Z0, Z1, NZ + 1)
    current, radial_moment, centres = [], [], []
    integral_current = integral_r = 0.0
    for a, b in zip(radial_edges[:-1], radial_edges[1:]):
        for c, d in zip(vertical_edges[:-1], vertical_edges[1:]):
            t0, t1 = a - R0, b - R0
            part = ALPHA * (t1 - t0) + BETA * (t1**2 - t0**2) / 2.0
            cell_current = (d - c) * part
            int_r = (d - c) * (
                R0 * part + ALPHA * (t1**2 - t0**2) / 2.0 + BETA * (t1**3 - t0**3) / 3.0
            )
            centre_r = (a + b) / 2.0
            current.append(cell_current)
            radial_moment.append(int_r - centre_r * cell_current)
            centres.append((centre_r, (c + d) / 2.0))
            integral_current += cell_current
            integral_r += int_r
    exact = (integral_r / integral_current, (Z0 + Z1) / 2.0)
    return (
        np.asarray(centres, dtype=np.float64),
        np.asarray(current, dtype=np.float64),
        np.asarray(radial_moment, dtype=np.float64),
        np.zeros(len(current), dtype=np.float64),
        exact,
    )


# --- more helpers ---


def _second_moment(count):
    """A positive-definite per-cell second-moment ``[radial, vertical, cross]``."""
    index = np.arange(count, dtype=np.float64)
    return np.column_stack(
        [0.30 + 0.01 * index, 0.20 + 0.004 * index, 0.03 * np.sin(1.7 * index)]
    )


def _couple(second, radial_moment, vertical_moment):
    """Forward coupling transform: divide by the second-moment matrix."""
    radial, vertical, cross = second[:, 0], second[:, 1], second[:, 2]
    determinant = radial * vertical - cross**2
    assert np.all(determinant > 0.0), "second-moment matrix must be positive definite"
    return (
        (vertical * radial_moment - cross * vertical_moment) / determinant,
        (-cross * radial_moment + radial * vertical_moment) / determinant,
    )


def _recovered_centroid(perturb):
    """Centroid of the bank after forward coupling and the benchmark helper inverse."""
    centres, current, radial, vertical, _ = _rectangle_distribution()
    second = _second_moment(len(centres))
    coupled_r, coupled_v = _couple(second, radial, vertical)
    recovered_r, recovered_v = recover_physical_first_moments(
        second * (1.0 + perturb), coupled_r, coupled_v
    )
    return current_centroid(centres, current, recovered_r, recovered_v)


def _direct_centroid():
    """Centroid of the bank from its own moments, before any transform."""
    centres, current, radial, vertical, _ = _rectangle_distribution()
    return current_centroid(centres, current, radial, vertical)


# --- tests ---


def test_centroid_round_trip_recovers_exact_centroid():
    """The coupling inverse recovers the exact rectangle centroid to 1e-12 m."""
    _, _, _, _, exact = _rectangle_distribution()
    perturb = float(os.environ.get("NOVA_CENTROID_INVERSE_PERTURB", "0.0"))
    centroid_r, centroid_z = _recovered_centroid(perturb)
    assert abs(centroid_r - exact[0]) < TOLERANCE
    assert abs(centroid_z - exact[1]) < TOLERANCE
    # The bank's own moments already sit on the integral centroid, so the round
    # trip is compared against a known answer rather than a self-consistent one.
    direct_r, direct_z = _direct_centroid()
    assert abs(direct_r - exact[0]) < TOLERANCE
    assert abs(direct_z - exact[1]) < TOLERANCE


def test_perturbed_inverse_is_detected():
    """A deliberate inverse perturbation moves the centroid past tolerance.

    The radial first moment is the one that carries the centroid, so the
    perturbation shifts R well past tolerance while Z, whose first moment is
    zero, stays on the exact mid-height.
    """
    _, _, _, _, exact = _rectangle_distribution()
    centroid_r, centroid_z = _recovered_centroid(PERTURBATION)
    assert abs(centroid_r - exact[0]) > TOLERANCE
    assert abs(centroid_z - exact[1]) < TOLERANCE
