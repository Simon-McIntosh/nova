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
from typing import TYPE_CHECKING

import numpy as np

from nova.biot.greens import MU0

if TYPE_CHECKING:
    import jax

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

    This is a diagnostic identity, not the observed side of a constraint. It
    assumes a large aspect ratio and a circular cross-section, and at the
    aspect and elongation ratios of a spherical tokamak the neglected
    :math:`a/R` and shaping terms are of order the bracket itself, so the
    number it returns states the field a circular ring of the given radii
    would require rather than the field the plasma's own current distribution
    implies. The exact magnetics-implied combination is a contour integral of
    the poloidal field; use that wherever the value is compared against a
    measurement.
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

    Like :func:`shafranov_vertical_field` this is a diagnostic identity and
    not the observed side of a constraint: the elongation correction reduces
    the leading geometric bias but leaves the neglected higher-order shaping
    and triangularity terms, so it is not the exact combination the external
    magnetics imply.
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


@dataclass(frozen=True)
class ShafranovContourIntegrals:
    r"""Exact virial contour integrals of the field on the plasma boundary.

    Every field is evaluated ONLY on the boundary contour, so each member is a
    line integral; the volume integral each one equals is stated in
    :func:`shafranov_contour_integrals`, and that equality is the gate.
    """

    radial_moment: jax.Array
    vertical_moment: jax.Array
    toroidal_stress: jax.Array
    perimeter: jax.Array
    enclosed_area: jax.Array


def _contour_metric(
    contour: jax.Array, count: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Return boundary points, segment lengths, outward normals and area.

    ``contour`` carries ``(N, 2)`` rows of :math:`(R, Z)` in metres, with rows
    at index ``count`` and beyond required to be exact zeros.  Segments close
    from the last live point back to the first, so the sampled boundary is a
    closed polyline.  The normal is the tangent rotated a quarter turn, then
    oriented outward by the sign of the shoelace area, which makes the
    result independent of the direction the caller sampled the contour in.
    """
    import jax.numpy as jnp

    points = contour[:count]
    following = jnp.concatenate((points[1:], points[:1]), axis=0)
    segment = following - points
    length = jnp.linalg.norm(segment, axis=-1)
    tangent = segment / jnp.where(length > 0.0, length, 1.0)[..., None]
    normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    signed_area = 0.5 * jnp.sum(
        points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    )
    normal = jnp.where(signed_area >= 0.0, normal, -normal)
    return points, length, normal, signed_area


def shafranov_contour_integrals(
    contour: jax.Array,
    radial_field: jax.Array,
    vertical_field: jax.Array,
    toroidal_field: jax.Array,
    count: int,
) -> ShafranovContourIntegrals:
    r"""Return the exact Shafranov virial integrals of a boundary contour.

    Derivation
    ----------
    In a static axisymmetric equilibrium the Maxwell-plus-pressure stress

    .. math::

       T = \frac{B \otimes B - \tfrac{1}{2} B^2 I}{\mu_0} - p I

    is divergence-free inside the plasma, :math:`\nabla \cdot T = 0`, with
    :math:`B = B_R e_R + B_\phi e_\phi + B_Z e_Z` and :math:`I` the identity.
    The divergence theorem applied to :math:`T \xi` for any smooth vector
    field :math:`\xi` gives the virial identity

    .. math::

       \int_\Omega T : \nabla \xi \, dV = \oint_C \xi \cdot T \cdot n \, dS,

    with :math:`\Omega` the plasma volume, :math:`C` its boundary, :math:`n`
    the outward unit normal in the :math:`(R, Z)` half-plane, :math:`dS =
    2\pi R \, dl` the toroidal surface element and :math:`dl` the arc length
    along :math:`C`.  Pressure vanishes on :math:`C` by definition of the last
    closed flux surface, so the contour side of every member below carries
    magnetic stresses only.

    Three fields :math:`\xi` give the three integrals returned here.

    * :math:`\xi = R e_R` (the Cartesian vector :math:`(x, y, 0)`, so
      :math:`\nabla \xi = \mathrm{diag}(1, 1, 0)`), whose volume side is
      :math:`T_{xx} + T_{yy} = -(B_Z^2/\mu_0 + 2p)`.  This is
      ``radial_moment``.
    * :math:`\xi = R e_R + Z e_Z = r`, the position vector, with
      :math:`\nabla \xi = I`, whose volume side is
      :math:`\mathrm{tr}\,T = -(B^2 / 2\mu_0 + 3p)`.  This is
      ``vertical_moment``.
    * :math:`\xi = e_R`, with :math:`\nabla e_R = e_\phi e_\phi / R`, whose
      volume side is :math:`T_{\phi\phi} / R =
      [(B_\phi^2 - B_p^2) / 2\mu_0 - p] / R`.  This is ``toroidal_stress``.

    All three include the toroidal field through :math:`B^2`, which is why
    ``toroidal_field`` is an argument: the poloidal field alone does not close
    the identities.

    Returns
    -------
    The three contour integrals in tesla-squared metres cubed per ampere
    (:math:`\mathrm{T^2 \, m^3 / A}`) for the two moments, per square metre for
    the stress, together with the contour perimeter [m] and its enclosed
    :math:`(R, Z)` area [m^2].  The moments are negative for a plasma whose
    pressure and field are positive, since both volume sides are negative
    there.

    Notes
    -----
    ``count`` must be a Python integer: it fixes the polyline shape, so the
    function stays traceable, differentiable and jit-able with respect to the
    field arguments.  ``MU0`` is the explicit permeability of free space and
    every symbol above is in raw SI.

    The combination :math:`\beta_p + l_i/2` is NOT a functional of these
    contour integrals alone.  Eliminating the toroidal term from
    ``vertical_moment`` needs the volume integral
    :math:`Z = \int B_\phi^2 \, dV`, and the pressure term needs
    :math:`P = \int p \, dV`; neither is recoverable from boundary data.  See
    :func:`beta_p_plus_half_internal_inductance` for the resulting assembly and
    its conditioning.
    """
    import jax.numpy as jnp

    points, length, normal, area = _contour_metric(contour, count)
    radial = radial_field[:count]
    vertical = vertical_field[:count]
    toroidal = toroidal_field[:count]
    poloidal_squared = radial**2 + vertical**2
    total_squared = poloidal_squared + toroidal**2
    inverse = 1.0 / MU0
    stress_rr = (radial**2 - 0.5 * total_squared) * inverse
    stress_zz = (vertical**2 - 0.5 * total_squared) * inverse
    stress_rz = radial * vertical * inverse
    radial_traction = stress_rr * normal[:, 0] + stress_rz * normal[:, 1]
    vertical_traction = stress_rz * normal[:, 0] + stress_zz * normal[:, 1]
    surface = 2.0 * jnp.pi * points[:, 0] * length
    return ShafranovContourIntegrals(
        radial_moment=jnp.sum(points[:, 0] * radial_traction * surface),
        vertical_moment=jnp.sum(
            (points[:, 0] * radial_traction + points[:, 1] * vertical_traction)
            * surface
        ),
        toroidal_stress=jnp.sum(radial_traction * surface),
        perimeter=jnp.sum(length),
        enclosed_area=area,
    )


def beta_p_plus_half_internal_inductance(
    integrals: ShafranovContourIntegrals,
    pressure_integral: jax.Array,
    toroidal_field_volume: jax.Array,
    major_radius: jax.Array,
    plasma_current: jax.Array,
) -> jax.Array:
    r"""Return :math:`\beta_p + l_i/2` in the volume-averaged normalisation.

    Matches :func:`nova.equilibrium.observation.observe_moments` exactly, whose
    targets are :math:`\beta_p = 4 P / (\mu_0 \bar R I_p^2)` and
    :math:`l_i = W / (\tfrac{1}{2} \mu_0^2 I_p^2 \bar R)` with
    :math:`P = \int p \, dV`, :math:`W = \int B_p^2 \, dV` and
    :math:`\bar R = \int R \, dV / \int dV` the volume-weighted major radius.
    Hence

    .. math::

       \beta_p + \frac{l_i}{2}
       = \frac{4 P + W / \mu_0}{\mu_0 \bar R I_p^2}.

    ``vertical_moment`` is :math:`-\int (B^2/2\mu_0 + 3p) \, dV`, so it fixes
    :math:`W` only against two further volume integrals:

    .. math::

       W = -2 \mu_0 \, S_v - 6 \mu_0 P - Z, \qquad
       Z = \int B_\phi^2 \, dV,

    giving

    .. math::

       \beta_p + \frac{l_i}{2}
       = \frac{-2 S_v - 2 P - Z / \mu_0}{\mu_0 \bar R I_p^2}.

    Parameters
    ----------
    integrals
        The contour integrals from :func:`shafranov_contour_integrals`.
    pressure_integral, toroidal_field_volume
        :math:`P` and :math:`Z` in joules and joules respectively
        (:math:`B^2/\mu_0` has units of pressure, so both are energies).
    major_radius, plasma_current
        :math:`\bar R` in metres and :math:`I_p` in amperes.

    Notes
    -----
    This assembly is exact in exact arithmetic and ill-conditioned in floating
    point.  :math:`S_v` and :math:`Z/\mu_0` are both of order
    :math:`(B^2/\mu_0) V`, while :math:`W/\mu_0`, the quantity their
    elimination leaves, is smaller by the square of the inverse aspect ratio:
    on the conventional-aspect-ratio single-null reference the two terms cancel
    to about three parts in ten thousand of the result.  A caller that wants a
    low-variance :math:`\beta_p + l_i/2` should integrate :math:`B_p^2` over
    the volume directly, as the clipped support measure already does; the
    contour route to :math:`W` is the one that amplifies boundary noise.
    """
    poloidal_field_volume = (
        -2.0 * MU0 * integrals.vertical_moment
        - 6.0 * MU0 * pressure_integral
        - toroidal_field_volume
    )
    return (4.0 * pressure_integral / MU0 + poloidal_field_volume / MU0**2) / (
        major_radius * plasma_current**2
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
    "ShafranovContourIntegrals",
    "VerticalConditioningReceipt",
    "beta_p_plus_half_internal_inductance",
    "decay_index",
    "shafranov_contour_integrals",
    "shafranov_vertical_field",
    "shafranov_vertical_field_elongated",
    "vertical_conditioning_receipt",
]
