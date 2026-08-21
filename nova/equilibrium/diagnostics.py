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

from dataclasses import dataclass

import numpy as np

from nova.biot.greens import MU0

#: Open stability interval for rigid vertical and radial displacements.
DECAY_INDEX_WINDOW: tuple[float, float] = (0.0, 1.5)


@dataclass(frozen=True)
class VerticalConditioningReceipt:
    r"""Conditioning of the axisymmetric vertical displacement mode.

    ``stability_margin`` is the signed distance to the nearest edge of the
    open vacuum decay-index interval.  Positive values lie inside the
    interval; zero and negative values are marginal or outside it.  The
    receipt reports conditioning only and never constrains a solve.
    """

    evaluation_radius_m: float
    vertical_field_t: float
    decay_index: float
    lower_stability_margin: float
    upper_stability_margin: float
    stability_margin: float
    stable: bool


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


def vertical_conditioning_receipt(
    radius: np.ndarray,
    vertical_field: np.ndarray,
    evaluation_radius: float,
) -> VerticalConditioningReceipt:
    r"""Return the local rigid-mode conditioning receipt at one radius.

    The decay index is evaluated on the supplied radial field sample and
    linearly interpolated at ``evaluation_radius``.  The radius must lie
    within the sampled interval.  A field null or non-finite local stencil
    produces NaN margins and a failing receipt instead of inventing a finite
    conditioning claim.
    """

    radial_coordinate = np.asarray(radius, dtype=np.float64)
    field = np.asarray(vertical_field, dtype=np.float64)
    index = decay_index(radial_coordinate, field)
    selected_radius = float(evaluation_radius)
    if not np.isfinite(selected_radius):
        raise ValueError("evaluation_radius must be finite")
    lower_radius = float(np.min(radial_coordinate))
    upper_radius = float(np.max(radial_coordinate))
    if not lower_radius <= selected_radius <= upper_radius:
        raise ValueError("evaluation_radius must lie within the radial sample")

    if radial_coordinate[0] > radial_coordinate[-1]:
        radial_coordinate = radial_coordinate[::-1]
        field = field[::-1]
        index = index[::-1]
    selected_field = float(np.interp(selected_radius, radial_coordinate, field))
    selected_index = float(np.interp(selected_radius, radial_coordinate, index))
    lower_bound, upper_bound = DECAY_INDEX_WINDOW
    lower_margin = selected_index - lower_bound
    upper_margin = upper_bound - selected_index
    margin = min(lower_margin, upper_margin)
    stable = bool(np.isfinite(margin) and margin > 0.0)
    return VerticalConditioningReceipt(
        evaluation_radius_m=selected_radius,
        vertical_field_t=selected_field,
        decay_index=selected_index,
        lower_stability_margin=float(lower_margin),
        upper_stability_margin=float(upper_margin),
        stability_margin=float(margin),
        stable=stable,
    )


__all__ = [
    "DECAY_INDEX_WINDOW",
    "VerticalConditioningReceipt",
    "decay_index",
    "shafranov_vertical_field",
    "shafranov_vertical_field_elongated",
    "vertical_conditioning_receipt",
]
