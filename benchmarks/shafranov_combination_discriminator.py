"""Attribute the Shafranov gap: EFIT's own value, the profiles, and the magnetics.

Each bank row carries a stored terminal state and the reconstruction group's own
plasma parameters, so ``beta_p + l_i/2`` can be read five ways on the same state
without running a solve:

* EFIT's own ``betap + li/2`` from the reconstruction group's scalars;
* the combination the row's extracted profiles imply, from
  :meth:`ForwardProfile.integral_observation`;
* the combination the external magnetics imply through the large-aspect-ratio
  vertical-field identity, with the minor radius the Shafranov logarithm is
  stated against taken three ways: the boundary's own half radial extent ``a``,
  the area-equivalent ``a * sqrt(kappa)``, and the production row's own kernel.

All three magnetics readings are the *same* identity.  The first two invert it
here, at two ln arguments; the third is
:meth:`ExternalShafranovConstraint.observed`, the production row's own
combination, which carries ``ln(8R/a)`` inside it.  None of the three is
formula-free, so their agreement is a consistency check between the row and the
diagnostic inversion, not an independent measurement of the quantity.  The
formula-free value of this combination is a contour integral of the poloidal
field over a contour enclosing the plasma; no reading here computes one.

What the table is for is attribution.  The gap between the profiles' reading of
``beta_p + l_i/2`` and the magnetics' reading is the quantity the constrained
row exists to close; placing EFIT's own value beside both says which side of
that gap the reconstruction sits on.  Which of the three magnetics readings sits
nearest EFIT's own sitting is a weaker statement than it looks: they differ only
in the ``ln(8R/a)`` argument, so a closure names the argument that happens to
match rather than a mechanism.

The unit and COCOS check of the profile-implied side is written out in full:
``beta_p`` and ``l_i`` are recomputed from the observation's own volume
integrals and compared with the observation's returned values, so the ``mu_0``
placement is measured rather than asserted.  Neither quantity carries a COCOS
sign factor, because each is a ratio in which the poloidal flux enters squared;
a convention that flips the sign of the flux flips the axis and the boundary
does not change either volume integral.  The magnetics side is signed through
``B_v``, so the identity inversion is stated with the plasma current's own sign.

One figure is written: a per-row strip of the five values on a shared axis, so a
reader sees on which side of the magnetics readings the reconstruction sits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import zarr

from benchmarks import settled_mask_stall as settled
from nova.biot.greens import MU0
from nova.equilibrium.constraint import (
    ConstraintContext,
    ExternalShafranovConstraint,
    sample_lattice_flux,
)
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.diagnostics import (
    shafranov_vertical_field,
    shafranov_vertical_field_elongated,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.equilibrium.wall_mask import WallUnit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = (
    ROOT / "docs/figures/constraint-augmented-newton-krylov/shafranov-discriminator"
)
#: Store group the reconstruction scalars and the terminal state are read from.
EFIT_GROUP = "efm"
#: Display raster resolution for the per-row state contour panels.
RASTER_SAMPLES = 181
#: Reading keys, in the order every row document and the panel order them.
READING_KEYS: tuple[str, ...] = (
    "efit_own",
    "profiles",
    "magnetics_circular",
    "magnetics_elongated",
    "magnetics_discrete",
)
READING_LABELS: dict[str, str] = {
    "efit_own": "EFIT betap + li/2",
    "profiles": "profiles",
    "magnetics_circular": "magnetics (a)",
    "magnetics_elongated": "magnetics (a sqrt k)",
    "magnetics_discrete": "magnetics (row kernel)",
}
#: What each magnetics reading is, so the labels cannot be read as independence.
MAGNETICS_PROVENANCE: dict[str, str] = {
    "magnetics_circular": (
        "identity_combination at a = half the boundary's radial extent: the "
        "large-aspect-ratio vertical-field identity, inverted here"
    ),
    "magnetics_elongated": (
        "the same identity inverted here at a*sqrt(kappa), the area-equivalent "
        "horizontal scale; only the ln argument differs"
    ),
    "magnetics_discrete": (
        "ExternalShafranovConstraint.observed, the production row's own "
        "combination, evaluated at the solved state's current centroid. The row "
        "carries ln(8R/a) inside its own expression, so the row's reading is the "
        "same identity, not a formula-free measurement. The formula-free value of "
        "this quantity is a contour integral of the poloidal field over a contour "
        "enclosing the plasma, which no reading in this driver computes"
    ),
}
#: Relative tolerance the analytic inversion is round-tripped against.
INVERSION_TOLERANCE = 1.0e-9
#: Field names of one row document, in the order the document is written.
ROW_FIELDS: tuple[str, ...] = (
    "identity",
    "efit_own_combination",
    "efit_poloidal_beta",
    "efit_internal_inductance",
    "profile_implied_combination",
    "magnetics_implied_combination",
    "minor_radius_m",
    "unnormalised_minor_radius_m",
    "elongation",
    "major_radius_m",
    "plasma_current_a",
    "vertical_field_t",
    "identity_round_trip_residual",
    "row_kernel_versus_circular_residual",
    "profile_implied_combination_unnormalised",
    "profile_normalisation",
    "constraint_row_reading_unnormalised",
    "constraint_row_normalisation",
    "commensurability",
    "convention_sentence",
    "state_panel",
    "readings",
    "sentence",
)


def _strict_float(value: Any) -> float | None:
    """Return a finite float or ``None`` so JSON carries no NaN."""
    if value is None:
        return None
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _relative_difference(left: float | None, right: float | None) -> float | None:
    """Return ``|left - right| / |right|``, or ``None`` where it is undefined."""
    if left is None or right is None:
        return None
    if not (np.isfinite(left) and np.isfinite(right)) or right == 0.0:
        return None
    return float(abs(left - right) / abs(right))


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def identity_combination(
    plasma_current: float,
    major_radius: float,
    minor_radius: float,
    vertical_field: float,
) -> float:
    r"""Invert the large-aspect-ratio identity for :math:`\beta_p + l_i/2`.

    The forward identity the diagnostic module evaluates is

    .. math::

       B_v = -\frac{\mu_0 I_p}{4 \pi R}
             \left[\ln\left(\frac{8R}{a}\right) + \beta_p + \frac{l_i}{2}
             - \frac{3}{2}\right],

    so the combination the field implies is

    .. math::

       \beta_p + \frac{l_i}{2}
         = -\frac{4 \pi R B_v}{\mu_0 I_p}
           - \ln\left(\frac{8R}{a}\right) + \frac{3}{2}.

    A non-finite input, a zero plasma current, or a minor radius outside the
    open interval ``(0, 8R)`` returns NaN, matching the forward identity.
    """
    current = float(plasma_current)
    radius = float(major_radius)
    minor = float(minor_radius)
    field = float(vertical_field)
    if not all(np.isfinite(value) for value in (current, radius, minor, field)):
        return float("nan")
    if current == 0.0 or radius <= 0.0 or minor <= 0.0 or minor >= 8.0 * radius:
        return float("nan")
    return (
        -4.0 * np.pi * radius * field / (MU0 * current)
        - float(np.log(8.0 * radius / minor))
        + 1.5
    )


def linear_combination(poloidal_beta: float, internal_inductance: float) -> float:
    """Return ``beta_p + l_i/2`` from the two scalar readings of a profile."""
    return float(poloidal_beta) + 0.5 * float(internal_inductance)


def boundary_shape(boundary: np.ndarray) -> tuple[float, float]:
    """Return the boundary's half radial extent [m] and its elongation.

    The half radial extent is the horizontal scale the Shafranov logarithm is
    stated against; the elongation is the half vertical extent over it, so the
    area-equivalent horizontal scale is ``a * sqrt(kappa)``.  Both are figures of
    the stored last closed flux surface, not operands chosen here.
    """
    points = np.asarray(boundary, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 3:
        raise ValueError("the stored boundary carries too few finite nodes")
    minor = 0.5 * float(np.ptp(points[:, 0]))
    vertical = 0.5 * float(np.ptp(points[:, 1]))
    if minor <= 0.0:
        raise ValueError("the stored boundary carries no radial extent")
    return minor, vertical / minor


def readings(combination_by_reading: dict[str, float | None]) -> dict[str, Any]:
    """Return the ordered readings block one row document carries."""
    return {
        key: {
            "label": READING_LABELS[key],
            "combination": _strict_float(combination_by_reading.get(key)),
        }
        for key in READING_KEYS
    }


def convention_clause(
    block: dict[str, Any], *, profile_combination: float | None
) -> str:
    """State what the stored moments' implied radius says about commensurability."""
    implied = block.get("implied_radius_from_betap_m")
    major = block.get("nova_major_radius_m")
    minor = block.get("boundary_minor_radius_m")
    rescaled = block.get("rescaled_efit_combination")
    if implied is None or major is None or minor is None:
        return (
            "The reconstruction's stored moments do not imply a denominator "
            "radius, so whether nova's own radius convention is commensurate "
            "with them is unstated for this row."
        )
    shared = block.get("implied_radius_inductance_over_betap")
    agreement = (
        f"the two stored scalars imply the same radius to within "
        f"{abs(shared - 1.0):.2%} (l_i implies "
        f"{block['implied_radius_from_inductance_m']:.4f} m), so one convention "
        f"difference explains both moments"
        if shared is not None
        else "l_i implies no finite radius, so only beta_p settles the scale"
    )
    return (
        f"Commensurability: on nova's own volume integral and leading factor, "
        f"the stored scalars are reproduced by a denominator radius of "
        f"{implied_label(block)} m, against nova's volume-weighted major radius "
        f"{major:.4f} m and the boundary's minor radius {minor:.4f} m; {agreement}. "
        f"Rescaled onto nova's radius the reconstruction's own combination is "
        f"{rescaled:.4f}"
        + (
            f", against the profiles' {profile_combination:.4f}."
            if profile_combination is not None and rescaled is not None
            else "."
        )
    )


def implied_label(block: dict[str, Any]) -> str:
    """Format the radius the stored moments imply."""
    value = block.get("implied_radius_from_betap_m")
    return "unstated" if value is None else f"{value:.4f}"


def attribute(
    combinations: dict[str, float | None],
) -> dict[str, Any]:
    """State which side EFIT sits on and which magnetics reading closes on it.

    The side is stated against the profiles' reading rather than against any
    reading's distance to EFIT, because the two sides of the gap are the
    profile-implied and magnetics-implied answers; the closing reading is the
    magnetics one whose combination sits nearest EFIT's own.
    """
    own = _strict_float(combinations.get("efit_own"))
    implied = _strict_float(combinations.get("profiles"))
    magnetics = {
        key: _strict_float(combinations.get(key))
        for key in ("magnetics_circular", "magnetics_elongated", "magnetics_discrete")
    }
    finite = {key: value for key, value in magnetics.items() if value is not None}
    if own is None or implied is None:
        side = "unstated: EFIT's own value or the profile-implied value is absent"
    elif own > implied:
        side = f"above the profiles' {implied:.4f}"
    elif own < implied:
        side = f"below the profiles' {implied:.4f}"
    else:
        side = "on the profiles' value"
    if not finite or own is None:
        closest = None
        sentence = (
            f"EFIT's own {own!r} sits {side}; no magnetics reading is available "
            "to close on it."
        )
    else:
        closest = min(finite, key=lambda key: abs(finite[key] - own))
        sentence = (
            f"EFIT's own {own:.4f} sits {side}; the closest magnetics reading is "
            f"{READING_LABELS[closest]} at {finite[closest]:.4f} "
            f"(residual {finite[closest] - own:+.4f}), against the magnetics set "
            + ", ".join(
                f"{READING_LABELS[key]} {value:.4f}"
                for key, value in magnetics.items()
                if value is not None
            )
            + "."
        )
    return {"side": side, "closing_reading": closest, "sentence": sentence}


def recomputed_profile_moments(observation) -> dict[str, Any]:
    r"""Recompute ``beta_p`` and ``l_i`` from the observation's volume integrals.

    This is the unit check of the profile-implied side.  The returned fields are
    compared against

    .. math::

       \beta_p = \frac{4 \int p\, dV}{\mu_0 R_{ax} I_p^2}, \qquad
       l_i = \frac{2 \int B_p^2\, dV}{\mu_0^2 R_{ax} I_p^2},

    evaluated on the observation's own ``pressure_integral``,
    ``poloidal_field_integral``, ``major_radius`` and ``plasma_current``, so a
    misplaced factor of ``mu_0`` or of the leading 4 or 2 is a stated relative
    difference rather than a silent convention.
    """
    current = float(np.asarray(observation.plasma_current))
    radius = float(np.asarray(observation.major_radius))
    pressure = float(np.asarray(observation.pressure_integral))
    field = float(np.asarray(observation.poloidal_field_integral))
    denominator = MU0 * radius * current**2
    beta = 4.0 * pressure / denominator
    inductance = 2.0 * field / (MU0 * denominator)
    reported_beta = float(np.asarray(observation.poloidal_beta))
    reported_inductance = float(np.asarray(observation.internal_inductance))

    def relative(left: float, right: float) -> float | None:
        return None if right == 0.0 else float(abs(left - right) / abs(right))

    return {
        "definition": (
            "beta_p = 4 * pressure_integral / (mu0 * major_radius * I_p**2); "
            "l_i = 2 * poloidal_field_integral / (mu0**2 * major_radius * I_p**2)"
        ),
        "mu0_h_m": MU0,
        "pressure_integral_j": pressure,
        "poloidal_field_integral_t2_m3": field,
        "major_radius_m": radius,
        "plasma_current_a": current,
        "recomputed_poloidal_beta": beta,
        "reported_poloidal_beta": reported_beta,
        "poloidal_beta_relative_difference": relative(beta, reported_beta),
        "recomputed_internal_inductance": inductance,
        "reported_internal_inductance": reported_inductance,
        "internal_inductance_relative_difference": relative(
            inductance, reported_inductance
        ),
        "cocos": (
            "neither quantity carries a sign factor: each is a ratio of a volume "
            "integral to mu0 * R_ax * I_p**2, and the poloidal field enters "
            "squared, so a COCOS convention that flips the sign of the poloidal "
            "flux leaves the pressure integral, the squared-field integral and "
            "the plasma current's own sign unchanged. The magnetics side is "
            "signed through B_v, so its inversion is stated with the plasma "
            "current's signed value."
        ),
    }


def _efit_scalars(group, row: int) -> dict[str, float]:
    """Return the reconstruction group's own beta_p and l_i at one slice."""

    def scalar(name: str) -> float:
        return float(np.asarray(group[name][...], dtype=float)[row])

    return {"betap": scalar("betap"), "li": scalar("li")}


def implied_radius(
    reported: float,
    integral: float,
    leading: float,
    mu0_power: float,
    plasma_current: float,
) -> float | None:
    r"""Return the radius a stored scalar is consistent with.

    nova's definitions are

    .. math::

       \beta_p = \frac{4 \int p\, dV}{\mu_0 R I_p^2}, \qquad
       l_i = \frac{2 \int B_p^2\, dV}{\mu_0^2 R I_p^2},

    so a stored scalar, evaluated on the same integral with the same leading
    factor, implies the denominator radius

    .. math::

       R_{\text{implied}}
         = \frac{\text{leading} \cdot \text{integral}}
                {\mu_0^{n} I_p^2 \cdot \text{reported}} .

    Asking the store for this radius is what makes the commensurability question
    answerable without assuming the reconstruction's convention: if the implied
    radius is nova's own volume-weighted major radius, the two conventions agree;
    if it is the boundary's minor radius, they differ by the aspect ratio, and
    the difference is a stated factor rather than an unexplained gap.
    """
    denominator = mu0_power * plasma_current**2 * reported
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    radius = leading * integral / denominator
    return float(radius) if np.isfinite(radius) and radius > 0.0 else None


def commensurability(
    *,
    efit: dict[str, float],
    unit_check: dict[str, Any],
    major_radius: float,
    minor_radius: float,
) -> dict[str, Any]:
    """Ask the reconstruction's stored moments which radius they normalise with."""
    current = float(unit_check["plasma_current_a"])
    pressure = float(unit_check["pressure_integral_j"])
    field = float(unit_check["poloidal_field_integral_t2_m3"])
    from_beta = implied_radius(efit["betap"], pressure, 4.0, MU0, current)
    from_inductance = implied_radius(efit["li"], field, 2.0, MU0**2, current)
    shared = (
        from_inductance / from_beta
        if from_beta not in (None, 0.0) and from_inductance is not None
        else None
    )
    major_over_minor = major_radius / minor_radius if minor_radius else None
    rescaled = (
        linear_combination(efit["betap"], efit["li"]) * from_beta / major_radius
        if from_beta is not None and major_radius
        else None
    )
    return {
        "nova_definition": unit_check["definition"],
        "nova_radius_convention": (
            "beta_p = 4*int(p dV)/(mu0*R*Ip**2) and "
            "l_i = 2*int(Bp**2 dV)/(mu0**2*R*Ip**2), with R the "
            "volume-weighted major radius sum(radial volume elements)/volume"
        ),
        "stored_betap": _strict_float(efit["betap"]),
        "stored_internal_inductance": _strict_float(efit["li"]),
        "plasma_current_a": _strict_float(current),
        "pressure_integral_j": _strict_float(pressure),
        "poloidal_field_integral_t2_m3": _strict_float(field),
        "nova_major_radius_m": _strict_float(major_radius),
        "boundary_minor_radius_m": _strict_float(minor_radius),
        "major_over_minor": _strict_float(major_over_minor),
        "implied_radius_from_betap_m": _strict_float(from_beta),
        "implied_radius_from_inductance_m": _strict_float(from_inductance),
        "implied_radius_inductance_over_betap": _strict_float(shared),
        "implied_over_major_from_betap": _strict_float(
            from_beta / major_radius if from_beta is not None and major_radius else None
        ),
        "implied_over_major_from_inductance": _strict_float(
            from_inductance / major_radius
            if from_inductance is not None and major_radius
            else None
        ),
        "implied_over_minor_from_betap": _strict_float(
            from_beta / minor_radius if from_beta is not None and minor_radius else None
        ),
        "rescaled_efit_combination": _strict_float(rescaled),
    }


def constraint_context(flux, target_current, *, requested_class=None):
    """Build the traced context a constraint's ``observed`` is called with.

    The context's fields are ``(flux, requested_class, target_current, shadow)``,
    and the two middle ones are both optional currents-or-classes, so a
    positional construction silently exchanges them: the plasma current lands in
    ``requested_class`` and the row is handed ``target_current=None``, which
    reads its moments on the unnormalised path and reports a different
    combination.  Naming every field at one constructor is what makes that
    exchange impossible to write, and the constructor is what the regression
    test exercises.
    """
    return ConstraintContext(
        flux=flux,
        requested_class=requested_class,
        target_current=target_current,
        shadow=None,
    )


def _metrics(
    profile, state, target_current: float, minor_radius: float
) -> dict[str, Any]:
    """Return the magnetics-side readings of one terminal state."""
    external = profile.operator.prescribed_current_field
    if external is None:
        raise RuntimeError("the row needs a prescribed conductor field")
    lattice = profile.lattice
    grid = jnp.asarray(external.flux())[: lattice.node_count].reshape(lattice.shape)
    observation = profile.current_moment_observation(
        state,
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=target_current,
    )
    radius = float(np.asarray(observation.centroid_r))
    current = float(np.asarray(observation.plasma_current))
    payload = (
        jnp.asarray(external.flux())[: lattice.node_count],
        jnp.asarray(minor_radius),
    )
    row = ExternalShafranovConstraint(minor_radius=jnp.asarray(minor_radius))
    discrete = float(
        np.asarray(
            row.observed(
                profile,
                constraint_context(state, target_current),
                payload,
            )
        ).reshape(-1)[0]
    )
    step = float(np.asarray(lattice.radial_step))
    point = jnp.asarray([radius, float(np.asarray(observation.centroid_z))])
    upper = sample_lattice_flux(lattice, grid, point + jnp.asarray([step, 0.0]))
    lower = sample_lattice_flux(lattice, grid, point - jnp.asarray([step, 0.0]))
    vertical_field = float(
        np.asarray((upper - lower) / (2.0 * step * TOTAL_FLUX_FACTOR * radius))
    )
    unnormalised = profile.current_moment_observation(
        state,
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=None,
    )
    discrete_unnormalised = float(
        np.asarray(
            row.observed(
                profile,
                constraint_context(state, None),
                payload,
            )
        ).reshape(-1)[0]
    )
    return {
        "major_radius_m": radius,
        "plasma_current_a": current,
        "vertical_field_t": vertical_field,
        "discrete": discrete,
        "discrete_unnormalised": discrete_unnormalised,
        "observation_plasma_current_a": current,
        "unnormalised_plasma_current_a": float(np.asarray(unnormalised.plasma_current)),
        "unnormalised_major_radius_m": float(np.asarray(unnormalised.centroid_r)),
        "payload": payload,
        "lattice": lattice,
        "grid": grid,
        "point": point,
        "step": step,
    }


def _round_trip(
    plasma_current: float,
    major_radius: float,
    minor_radius: float,
    vertical_field: float,
    combination: float,
    *,
    elongation: float | None = None,
) -> float | None:
    """Return the relative residual of the analytic inversion's round trip.

    The forward evaluation uses the diagnostic module's own identity, so the
    residual measures the inversion against the function the row cites rather
    than against a second copy of the algebra.  A stated tolerance on this
    residual is what makes the analytic reading a measurement of the identity
    rather than an unchecked rearrangement of it.
    """
    if not np.isfinite(combination):
        return None
    forward = (
        shafranov_vertical_field(
            plasma_current, major_radius, minor_radius, combination
        )
        if elongation is None
        else shafranov_vertical_field_elongated(
            plasma_current, major_radius, minor_radius, elongation, combination
        )
    )
    if not np.isfinite(forward) or vertical_field == 0.0:
        return None
    return float(abs(forward - vertical_field) / abs(vertical_field))


def _round_trip_alias(
    shape: dict[str, Any], minor: float, circular: float, *, elongation: float | None
) -> float | None:
    """Return the circular inversion's round-trip residual from a metrics block."""
    return _round_trip(
        shape["plasma_current_a"],
        shape["major_radius_m"],
        minor,
        shape["vertical_field_t"],
        circular,
        elongation=elongation,
    )


def _row_document(
    profile,
    *,
    identity: str,
    state,
    target_current: float,
    efit: dict[str, float],
    boundary: np.ndarray,
) -> dict[str, Any]:
    """Read one bank row five ways and state the attribution."""
    minor, elongation = boundary_shape(boundary)
    shape = _metrics(profile, state, target_current, minor)
    observation = profile.integral_observation(state, target_current)
    unnormalised_observation = profile.integral_observation(state)
    profile_combination = linear_combination(
        np.asarray(observation.poloidal_beta),
        np.asarray(observation.internal_inductance),
    )
    unnormalised_profile_combination = linear_combination(
        np.asarray(unnormalised_observation.poloidal_beta),
        np.asarray(unnormalised_observation.internal_inductance),
    )
    unit_check = recomputed_profile_moments(observation)
    circular = identity_combination(
        shape["plasma_current_a"],
        shape["major_radius_m"],
        minor,
        shape["vertical_field_t"],
    )
    elongated = identity_combination(
        shape["plasma_current_a"],
        shape["major_radius_m"],
        minor * float(np.sqrt(elongation)),
        shape["vertical_field_t"],
    )
    combinations = {
        "efit_own": linear_combination(efit["betap"], efit["li"]),
        "profiles": profile_combination,
        "magnetics_circular": circular,
        "magnetics_elongated": elongated,
        "magnetics_discrete": shape["discrete"],
    }
    profile_current = _strict_float(observation.plasma_current)
    profile_current_difference = _relative_difference(
        profile_current, _strict_float(target_current)
    )
    if not bool(profile.operator.use_linear_moments):
        profile_path = (
            "unnormalised: the operator's moment path resolves moments from the "
            "flux directly and ignores the requested current, so the profile "
            "column is the unnormalised observation"
        )
    elif profile_current_difference == 0.0:
        profile_path = (
            "normalised: the observation's own current equals the requested "
            "current, so the requested normalisation is the identity"
        )
    elif profile_current_difference is None:
        profile_path = (
            "unstated: the observation's own current or the requested current is "
            "not finite, so the normalisation path cannot be read"
        )
    else:
        profile_path = (
            "normalised: the observation's own current differs from the requested "
            "current by a stated relative difference, so the profile column is "
            "the current-normalised observation"
        )
    profile_normalisation = {
        "requested_current_a": _strict_float(target_current),
        "observation_plasma_current_a": profile_current,
        "observation_relative_difference": profile_current_difference,
        "operator_use_linear_moments": bool(profile.operator.use_linear_moments),
        "path": profile_path,
        "normalised_combination": _strict_float(profile_combination),
        "unnormalised_combination": _strict_float(unnormalised_profile_combination),
        "combination_shift": _strict_float(
            profile_combination - unnormalised_profile_combination
        ),
    }
    convention = commensurability(
        efit=efit,
        unit_check=unit_check,
        major_radius=shape["major_radius_m"],
        minor_radius=minor,
    )
    constraint_normalisation = {
        "requested_current_a": _strict_float(target_current),
        "observed_current_a": _strict_float(shape["observation_plasma_current_a"]),
        "unnormalised_current_a": _strict_float(shape["unnormalised_plasma_current_a"]),
        "current_relative_difference": _relative_difference(
            _strict_float(shape["observation_plasma_current_a"]),
            _strict_float(shape["unnormalised_plasma_current_a"]),
        ),
        "observed_reading": _strict_float(shape["discrete"]),
        "unnormalised_reading": _strict_float(shape["discrete_unnormalised"]),
        "reading_shift": _strict_float(
            shape["discrete"] - shape["discrete_unnormalised"]
        ),
    }
    return {
        "identity": identity,
        "efit_own_combination": _strict_float(combinations["efit_own"]),
        "efit_poloidal_beta": _strict_float(efit["betap"]),
        "efit_internal_inductance": _strict_float(efit["li"]),
        "profile_implied_combination": _strict_float(profile_combination),
        "magnetics_implied_combination": _strict_float(shape["discrete"]),
        "minor_radius_m": _strict_float(minor),
        "unnormalised_minor_radius_m": _strict_float(minor),
        "elongation": _strict_float(elongation),
        "major_radius_m": _strict_float(shape["major_radius_m"]),
        "plasma_current_a": _strict_float(shape["plasma_current_a"]),
        "vertical_field_t": _strict_float(shape["vertical_field_t"]),
        "identity_round_trip_residual": _round_trip_alias(
            shape, minor, circular, elongation=None
        ),
        "identity_round_trip_residual_elongated": _round_trip(
            shape["plasma_current_a"],
            shape["major_radius_m"],
            minor,
            shape["vertical_field_t"],
            elongated,
            elongation=elongation,
        ),
        "row_kernel_versus_circular_residual": _strict_float(
            shape["discrete"] - circular
        ),
        "inversion_tolerance": INVERSION_TOLERANCE,
        "commensurability": convention,
        "convention_sentence": convention_clause(
            convention, profile_combination=profile_combination
        ),
        "readings": readings(combinations),
        "unit_check": unit_check,
        "sentence": attribute(combinations)["sentence"],
        "profile_implied_combination_unnormalised": _strict_float(
            unnormalised_profile_combination
        ),
        "profile_normalisation": profile_normalisation,
        "constraint_row_reading_unnormalised": _strict_float(
            shape["discrete_unnormalised"]
        ),
        "constraint_row_normalisation": constraint_normalisation,
    }


def _state_topology(operator, state) -> dict[str, Any]:
    """Return the state's read nulls, or a recorded refusal."""
    try:
        _masks, topology = operator.read(jnp.asarray(state))
    except NoQualifiedAxisError:
        return {"read_status": "no_qualified_axis"}
    diverted = bool(np.asarray(topology.diverted))
    return {
        "read_status": "qualified",
        "class": str(TopologyClass.DIVERTED if diverted else TopologyClass.LIMITED),
        "axis_rz_m": np.asarray(topology.axis, dtype=float).reshape(-1)[:2].tolist(),
        "x_point_rz_m": np.asarray(topology.x_point, dtype=float)
        .reshape(-1, 2)
        .tolist(),
    }


def _wall_units(operator) -> tuple[WallUnit, ...]:
    """Return the operator's wall as its own typed units.

    The wall is stored flat with unit offsets and per-unit closure and kind, so
    a panel draws every unit on its own terms rather than one invented ring.
    """
    coordinate = np.asarray(operator.wall.coordinate, dtype=float).reshape(-1, 2)
    offsets = np.asarray(operator.wall_unit_offsets, dtype=int)
    closed = np.asarray(operator.wall_unit_closed, dtype=bool)
    kinds = tuple(operator.wall_unit_kinds)
    return tuple(
        WallUnit(
            coordinate[start:stop, 0],
            coordinate[start:stop, 1],
            kind=kinds[index],
            closed=bool(closed[index]),
        )
        for index, (start, stop) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        )
    )


def _state_raster(profile, state, units, *, samples: int = RASTER_SAMPLES):
    """Interpolate one banked state onto a display raster for line contours."""
    points = np.asarray(profile.lattice.coordinate, dtype=float)
    field = np.asarray(state, dtype=float).reshape(-1)[: points.shape[0]]
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(field)
    points, field = points[finite], field[finite]
    if points.shape[0] < 3:
        raise ValueError("the state carries too few finite samples to contour")
    limits = np.vstack(
        (points, *[np.asarray(unit.vertices, dtype=float) for unit in units])
    )
    radial = np.linspace(
        float(np.min(limits[:, 0])), float(np.max(limits[:, 0])), samples
    )
    height = np.linspace(
        float(np.min(limits[:, 1])), float(np.max(limits[:, 1])), samples
    )
    radius_grid, height_grid = np.meshgrid(radial, height)
    raster = LinearNDInterpolator(points, field, fill_value=np.nan)(
        radius_grid, height_grid
    )
    return radial, height, np.asarray(raster, dtype=float)


def _render_state_panel(
    profile,
    state,
    *,
    path: Path,
    title: str,
    note: str,
) -> dict[str, Any]:
    """Draw one banked terminal state as shared-level line contours.

    The panel follows the project's plotting rules: unfilled contours on one
    physical level array so two panels cannot hide a mismatch behind their own
    colour scales, the state's stationary points marked in the ``draw_nulls``
    vocabulary, the vessel drawn unit-faithfully, and no axes or grid.
    """
    units = _wall_units(profile.operator)
    radial, height, field = _state_raster(profile, state, units)
    levels = poloidal.contour_levels(field, count=12)
    topology = _state_topology(profile.operator, state)
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    poloidal.draw_flux_contours(axis, radial, height, field, levels)
    poloidal.draw_wall(axis, units=units)
    if topology.get("read_status") == "qualified":
        poloidal.draw_nulls(
            axis,
            magnetic_axis=topology["axis_rz_m"],
            x_points=np.asarray(topology["x_point_rz_m"], dtype=float),
            style=DEFAULT_INK,
            contain=units,
        )
    poloidal_axes(axis)
    axis.set_title(f"{title}\n{note}", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)
    return {
        "png": {
            "filesystem_path": str(path),
            "project_absolute_src": (
                "/nova/figures/constraint-augmented-newton-krylov/"
                f"shafranov-discriminator/{path.name}"
            ),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "svg": {
            "filesystem_path": str(path.with_suffix(".svg")),
            "project_absolute_src": (
                "/nova/figures/constraint-augmented-newton-krylov/"
                f"shafranov-discriminator/{path.with_suffix('.svg').name}"
            ),
            "sha256": hashlib.sha256(path.with_suffix(".svg").read_bytes()).hexdigest(),
        },
        "levels_wb": [float(level) for level in levels],
        "topology": topology,
        "wall_unit_count": len(units),
    }


def _draw_panel(receipt: dict[str, Any], path: Path, *, source: str) -> dict[str, Any]:
    """Draw one strip per row, all five readings on a shared axis.

    A strip is the right form for this measurement: the readings are five
    scalars per row on one physical scale, and what a reader needs to see is
    their order along that scale, not a spatial or sequential relationship.  The
    readings share one axis inside each strip so a reader cannot compare two
    rows across independent scales.
    """
    rows = receipt["rows_receipt"]
    palette = {
        "efit_own": "#111111",
        "profiles": "#8c2d04",
        "magnetics_circular": "#3366cc",
        "magnetics_elongated": "#2a9d8f",
        "magnetics_discrete": "#cc7722",
    }
    markers = {
        "efit_own": "D",
        "profiles": "o",
        "magnetics_circular": "v",
        "magnetics_elongated": "^",
        "magnetics_discrete": "s",
    }
    figure, axes = plt.subplots(
        len(rows),
        1,
        figsize=(7.2, 0.95 * len(rows) + 1.0),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    spans = [
        value
        for row in rows
        for value in (row["readings"][key]["combination"] for key in READING_KEYS)
        if value is not None
    ]
    low, high = min(spans), max(spans)
    pad = 0.08 * (high - low if high > low else 1.0)
    for axis, row in zip(axes, rows, strict=True):
        for key in READING_KEYS:
            value = row["readings"][key]["combination"]
            if value is None:
                continue
            axis.plot(
                [value],
                [0.0],
                marker=markers[key],
                color=palette[key],
                markersize=7,
                linestyle="none",
                label=READING_LABELS[key] if axis is axes[0] else None,
            )
        profiles = row["readings"]["profiles"]["combination"]
        own = row["readings"]["efit_own"]["combination"]
        if profiles is not None:
            axis.axvline(profiles, color="#8c2d04", linewidth=0.8, alpha=0.4)
        if own is not None:
            axis.axvline(own, color="#111111", linewidth=0.8, alpha=0.4)
        axis.set_yticks([])
        axis.set_ylim(-0.5, 0.5)
        axis.set_title(
            f"{row['identity']}: a = {row['minor_radius_m']:.4f} m, "
            f"kappa = {row['elongation']:.3f}, "
            f"B_v = {row['vertical_field_t']:.3e} T",
            fontsize=8,
            loc="left",
        )
    axes[0].legend(fontsize=6, ncol=5, loc="lower center", frameon=False)
    axes[-1].set_xlabel(r"$\beta_p + l_i/2$", fontsize=8)
    axes[-1].set_xlim(low - pad, high + pad)
    figure.suptitle(
        "beta_p + l_i/2 on the MAST bank rows: EFIT's own value, the extracted "
        "profiles, and the magnetics read three ways",
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "filesystem_path": str(path),
        "project_absolute_src": (
            "/nova/figures/constraint-augmented-newton-krylov/"
            f"shafranov-discriminator/{path.name}"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_revision": source,
    }


def commensurability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """State the convention reading across the rows, from their own implied radii."""
    blocks = [
        (row["identity"], row["commensurability"])
        for row in rows
        if row.get("commensurability") is not None
    ]
    if not blocks:
        return {"reading": "no row carries a commensurability block"}
    implied = {
        identity: block["implied_over_major_from_betap"] for identity, block in blocks
    }
    return {
        "questions": (
            "whether the reconstruction's stored betap and li use the same "
            "volume-weighted major radius and the same l_i convention nova's "
            "definitions use, computed from the reconstruction's own stored "
            "scalars and nova's own volume integrals rather than asserted"
        ),
        "resolution": (
            "the stored scalars are evaluated on nova's own volume integrals and "
            "leading factors; the only free parameter left for the row is the "
            "denominator radius, and the radius that reproduces the store is "
            "compared with nova's own volume-weighted major radius and with the "
            "boundary's minor radius"
        ),
        "implied_over_major_from_betap": implied,
        "implied_radius_inductance_over_betap": {
            identity: block["implied_radius_inductance_over_betap"]
            for identity, block in blocks
        },
        "major_over_minor": {
            identity: block["major_over_minor"] for identity, block in blocks
        },
        "rescaled_efit_combination": {
            identity: block["rescaled_efit_combination"] for identity, block in blocks
        },
        "readings": [row["convention_sentence"] for row in rows],
    }


def measure(*, directory: Path, cache_root: Path | None = None) -> dict[str, Any]:
    """Read every bank row of the decomposition bank five ways."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    directory.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "receipt": (
            "beta_p + l_i/2 read five ways per bank row: EFIT's own scalars, the "
            "extracted profiles, and the external magnetics through the "
            "large-aspect-ratio identity at three minor radii"
        ),
        "row_set": (
            "every row the decomposition bank qualifies; the receipt records the "
            "full selection rather than a stated count"
        ),
        "solve_policy": "no new solve: every reading is taken on the banked state",
        "rows": [list(key) for key in sorted(selected)],
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "efit_scalars": f"{EFIT_GROUP}/betap and {EFIT_GROUP}/li at the row slice",
            "efit_combination": "betap + li/2 from the reconstruction's own scalars",
            "profile_combination": "poloidal_beta + internal_inductance/2 from "
            "ForwardProfile.integral_observation on the banked state",
            "magnetics_instrument": (
                "the prescribed conductor image on the lattice, sampled at the "
                "solved state's current centroid"
            ),
            "magnetics_circular": "identity_combination with a = half radial extent",
            "magnetics_elongated": "identity_combination with a * sqrt(kappa)",
            "magnetics_discrete": (
                "ExternalShafranovConstraint.observed: the production row's own "
                "combination, evaluated with the traced context's plasma current"
            ),
            "magnetics_provenance": dict(MAGNETICS_PROVENANCE),
            "formula_free_value": (
                "not computed here: the formula-free reading of this quantity is a "
                "contour integral of the poloidal field over a contour enclosing "
                "the plasma, which no driver in this receipt performs"
            ),
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"carrier_evidence": carrier_evidence},
        "rows_receipt": [],
    }
    for shot, row_index in sorted(selected):
        selected_row, qualification = selected[(shot, row_index)]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, _policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        state = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        group = zarr.open_group(str(settled.SHOT_STORE / f"{shot}.zarr"), mode="r")[
            EFIT_GROUP
        ]
        entry = _row_document(
            profile,
            identity=f"{shot}/{row_index}",
            state=state,
            target_current=target_current,
            efit=_efit_scalars(group, row_index),
            boundary=np.asarray(case["boundary"], dtype=float),
        )
        entry["state_panel"] = _render_state_panel(
            profile,
            state,
            path=directory / f"row-{shot}-{row_index}-state.png",
            title=(
                f"{shot}/{row_index}: banked terminal state, "
                f"kappa = {entry['elongation']:.3f}"
            ),
            note=(
                f"profiles {entry['profile_implied_combination']:.4f}, "
                f"EFIT {entry['efit_own_combination']:.4f}, "
                f"magnetics {entry['magnetics_implied_combination']:.4f}"
            ),
        )
        receipt["rows_receipt"].append(entry)
        write_row(directory, entry)
        print(
            "SHAFRANOV-DISCRIMINATOR " + json.dumps(entry, sort_keys=True), flush=True
        )
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    receipt["commensurability"] = commensurability_summary(receipt["rows_receipt"])
    print(
        "SHAFRANOV-COMMENSURABILITY "
        + json.dumps(receipt["commensurability"], sort_keys=True),
        flush=True,
    )
    receipt["figure"] = _draw_panel(
        receipt,
        directory / "shafranov-combination-discriminator.png",
        source=receipt["source"]["revision"],
    )
    (directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print("SHAFRANOV-DISCRIMINATOR-DONE", flush=True)
    return receipt


def write_row(directory: Path, entry: dict[str, Any]) -> Path:
    """Write one row document beside the panel."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"row-{entry['identity'].replace('/', '-')}.json"
    path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> None:
    """Run the combination discriminator from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--cache-root", type=Path, default=None)
    arguments = parser.parse_args(argv)
    measure(directory=arguments.directory, cache_root=arguments.cache_root)


if __name__ == "__main__":
    main()
