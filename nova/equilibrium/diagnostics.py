r"""Analytic radial force-balance diagnostics for axisymmetric equilibria.

The Shafranov identities return the signed vertical field required to balance
the outward hoop force of a large-aspect-ratio plasma current ring. Under
Nova's cylindrical convention, positive toroidal plasma current requires
negative :math:`B_z` so that :math:`I_p B_z` points radially inward.

The vacuum decay index measures the radial variation of that vertical field.
All quantities are raw SI, and the permeability convention is shared with the
canonical axisymmetric kernels in :mod:`nova.biot.greens`.
"""

from __future__ import annotations

import numpy as np

from nova.biot.greens import MU0

#: Open stability interval for rigid vertical and radial displacements.
DECAY_INDEX_WINDOW: tuple[float, float] = (0.0, 1.5)


def shafranov_vertical_field(
    plasma_current: float,
    major_radius: float,
    minor_radius: float,
    poloidal_beta_plus_half_internal_inductance: float,
) -> float:
    r"""Return the signed vertical field required by a circular current ring.

    The large-aspect-ratio identity is

    .. math::

       B_v = -\frac{\mu_0 I_p}{4\pi R}
             \left[\ln\left(\frac{8R}{a}\right)
             + \beta_p + \frac{l_i}{2} - \frac{3}{2}\right].

    Parameters are the plasma current [A], major and minor radii [m], and the
    dimensionless combination :math:`\beta_p + l_i/2`. Invalid ring geometry
    returns NaN. The result changes sign exactly when the plasma current does.
    """
    current = float(plasma_current)
    radius = float(major_radius)
    minor = float(minor_radius)
    if radius <= 0.0 or minor <= 0.0 or minor >= 8.0 * radius:
        return float("nan")
    force_factor = (
        np.log(8.0 * radius / minor)
        + float(poloidal_beta_plus_half_internal_inductance)
        - 1.5
    )
    return float(-MU0 * current / (4.0 * np.pi * radius) * force_factor)


def shafranov_vertical_field_elongated(
    plasma_current: float,
    major_radius: float,
    minor_radius: float,
    elongation: float,
    poloidal_beta_plus_half_internal_inductance: float,
) -> float:
    r"""Return the elongation-corrected signed vertical-field requirement.

    To leading order, an elliptical column replaces the circular minor radius
    by its area-equivalent value :math:`a\sqrt{\kappa}`. Thus the logarithmic
    term is :math:`\ln(8R/(a\sqrt{\kappa}))`. Unit elongation reproduces
    :func:`shafranov_vertical_field` exactly. Non-positive or non-finite
    elongation returns NaN.
    """
    shape = float(elongation)
    if shape <= 0.0 or not np.isfinite(shape):
        return float("nan")
    return shafranov_vertical_field(
        plasma_current,
        major_radius,
        float(minor_radius) * float(np.sqrt(shape)),
        poloidal_beta_plus_half_internal_inductance,
    )


def decay_index(radius: np.ndarray, vertical_field: np.ndarray) -> np.ndarray:
    r"""Return the vacuum decay index along a radial field sample.

    The index is

    .. math::

       n(R) = -\frac{R}{B_z}\frac{\partial B_z}{\partial R}.

    ``radius`` must be monotonic and have the same shape as ``vertical_field``.
    The derivative uses :func:`numpy.gradient`, including its behavior for
    undersized or inconsistent arrays. A field null has no meaningful decay
    index, so values within an amplitude-scaled zero tolerance return NaN.
    """
    radial_coordinate = np.asarray(radius, dtype=np.float64)
    field = np.asarray(vertical_field, dtype=np.float64)
    field_gradient = np.gradient(field, radial_coordinate)
    zero_tolerance = 1e-12 + 1e-6 * float(np.nanmax(np.abs(field), initial=0.0))
    index = np.where(
        np.abs(field) > zero_tolerance,
        -(radial_coordinate / field) * field_gradient,
        np.nan,
    )
    return np.asarray(index, dtype=np.float64)


__all__ = [
    "DECAY_INDEX_WINDOW",
    "decay_index",
    "shafranov_vertical_field",
    "shafranov_vertical_field_elongated",
]
