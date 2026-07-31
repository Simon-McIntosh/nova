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
    profile_term = float(poloidal_beta_plus_half_internal_inductance)
    if not all(np.isfinite(value) for value in (current, radius, minor, profile_term)):
        return float("nan")
    if radius <= 0.0 or minor <= 0.0 or minor >= 8.0 * radius:
        return float("nan")
    force_factor = np.log(8.0 * radius / minor) + profile_term - 1.5
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
    current = float(plasma_current)
    radius = float(major_radius)
    minor = float(minor_radius)
    shape = float(elongation)
    profile_term = float(poloidal_beta_plus_half_internal_inductance)
    if not all(
        np.isfinite(value) for value in (current, radius, minor, shape, profile_term)
    ):
        return float("nan")
    if shape <= 0.0:
        return float("nan")
    return shafranov_vertical_field(
        current,
        radius,
        minor * float(np.sqrt(shape)),
        profile_term,
    )


def decay_index(radius: np.ndarray, vertical_field: np.ndarray) -> np.ndarray:
    r"""Return the vacuum decay index along a radial field sample.

    The index is

    .. math::

       n(R) = -\frac{R}{B_z}\frac{\partial B_z}{\partial R}.

    Both inputs must be equal-shaped one-dimensional arrays with at least two
    samples. ``radius`` must be finite and strictly monotonic, increasing or
    decreasing. A field null has no meaningful decay index, so values within
    an amplitude-scaled zero tolerance return NaN. Non-finite field samples
    propagate through the local finite-difference stencil; a field with no
    finite samples returns all NaN.
    """
    radial_coordinate = np.asarray(radius, dtype=np.float64)
    field = np.asarray(vertical_field, dtype=np.float64)
    if radial_coordinate.ndim != 1 or field.ndim != 1:
        raise ValueError("radius and vertical_field must be one-dimensional")
    if radial_coordinate.shape != field.shape:
        raise ValueError("radius and vertical_field must have equal shapes")
    if radial_coordinate.size < 2:
        raise ValueError("radius and vertical_field must contain at least two samples")
    if not np.isfinite(radial_coordinate).all():
        raise ValueError("radius must contain only finite values")
    spacing = np.diff(radial_coordinate)
    if not (np.all(spacing > 0.0) or np.all(spacing < 0.0)):
        raise ValueError("radius must be strictly monotonic")

    finite_field = np.abs(field[np.isfinite(field)])
    if finite_field.size == 0:
        return np.full(field.shape, np.nan, dtype=np.float64)
    zero_tolerance = 1e-12 + 1e-6 * float(np.max(finite_field))
    with np.errstate(divide="ignore", invalid="ignore"):
        field_gradient = np.gradient(field, radial_coordinate)
        index = np.where(
            np.isfinite(field)
            & np.isfinite(field_gradient)
            & (np.abs(field) > zero_tolerance),
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
