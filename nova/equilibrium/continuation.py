r"""Bounded continuation of the source flux functions beyond the separatrix.

The core closure stops at the last closed flux surface. Continuing it is not
the extrapolation of a fitted curve: it is a second declared closure, on a
domain with different material connectivity, and everything a reader needs to
judge it — where it is anchored, how smoothly, on what functional form and out
to what bound — is declared rather than inferred.

Two branches, one distance
--------------------------
Normalised flux is not the continuation variable, because the two open
branches lie on opposite sides of it. The common scrape-off layer carries
:math:`\psi_N > 1`, while the private-flux region the X-point cut isolates
carries :math:`\psi_N < 1` exactly like the core. Both are continued in the
separatrix distance

.. math::
    d = \max\left[\, \sigma\,(\psi_N - 1),\ 0 \,\right], \qquad
    \sigma = +1 \ \text{(common SOL)}, \quad
    \sigma = -1 \ \text{(private flux)},

so one taper serves either branch and the sign is the only thing that
differs. With :math:`\psi_N = 1 + \sigma d` the chain rule gives
:math:`\mathrm{d}^k/\mathrm{d}\psi_N^k = \sigma^k \mathrm{d}^k/\mathrm{d}d^k`,
so a taper whose :math:`k`-th :math:`d`-derivative at the separatrix is
:math:`\sigma^k g^{(k)}(1)` continues the core gradient :math:`g` with
continuous :math:`\psi_N` derivatives on whichever branch it was declared for.
A continuation built for one branch is therefore wrong on the other, which is
why the domain is part of the declaration and is checked against the argument
it is supplied through.

Continuity is enforced, not hoped for
-------------------------------------
:class:`~nova.equilibrium.source.SeparatrixContinuity` states how many orders
the continuation matches, and its member value IS that count. Matching the
value alone leaves a gradient jump, matching nothing leaves a value jump, and
each of those jumps is a physical object: a jump in :math:`p'` or
:math:`FF'` across a flux surface is carried by a current sheet on that
surface, and a jump in their first derivative by a thin current layer. No
sheet, layer, sheath or halo model is declared in this solve — field-aligned
and poloidal scrape-off current wait on a specified material return path — so
declaring either class raises :class:`SeparatrixJumpError` and names the model
that would have to own it. Value and first-derivative matching is the minimum
admissible class.

The anchor is taken from the core closure by differentiating it at the
separatrix, and it is cross-checked against a one-sided difference from inside
the core. A core gradient with a kink there has no unambiguous separatrix
derivative, so the check rejects it rather than silently choosing a side.

Bounded support
---------------
Both families are bounded: beyond the declared outer support the source is
exactly zero, and no arithmetic on a cell past that bound contributes to any
integral. The families differ only in what they do at the bound.
:attr:`~nova.equilibrium.source.ContinuationForm.HERMITE_POLYNOMIAL` is the
minimal-degree polynomial that matches the declared orders at the separatrix
and vanishes to the same order at the bound, so the source has no step
anywhere.
:attr:`~nova.equilibrium.source.ContinuationForm.EXPONENTIAL_DECAY` carries a
declared e-folding width in separatrix distance, matches the same orders
through a polynomial prefactor, and is truncated at the bound; the receipt
publishes the amplitude that truncation discarded, so a support chosen too
tight for the declared width is visible rather than hidden.

Both families integrate in closed form, so the pressure and toroidal-field
primitives on an open domain carry no quadrature error and are constant beyond
the support bound, which is what a vanishing gradient there demands.

What makes the private-flux policy independent
----------------------------------------------
The private-flux branch is declared separately from the common scrape-off
layer and is neither copied from it nor silently zeroed. Its freedom is in the
form, the width and above all the support: the value and slope it starts from
are the core's, because anything else is a jump on a surface the two branches
share. A low-pressure private-flux region is therefore declared by giving it a
SHORT support — the pressure it carries stays within
:math:`|\Phi_b - \Phi_a| \int_0^{L} p'` of the boundary value, so shrinking
:math:`L` is what makes the region cold, and it does so without inventing a
discontinuity at the separatrix.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.convention import (
    flux_function_pressure,
    flux_function_toroidal_field,
)
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.source import (
    ContinuationForm,
    ContinuationRecord,
    DomainProfile,
    RotationClosure,
    SeparatrixContinuity,
    _validate_flux_function,
)

__all__ = [
    "ContinuedDomainProfile",
    "SeparatrixContinuation",
    "SeparatrixJumpError",
    "separatrix_derivatives",
]

#: Direction the separatrix distance grows in normalised flux, per open domain.
OUTWARD_SENSE: dict[PlasmaDomain, float] = {
    PlasmaDomain.COMMON_SOL: 1.0,
    PlasmaDomain.PRIVATE_FLUX: -1.0,
}

#: Step in normalised flux the one-sided check of a separatrix derivative is
#: taken over, and the relative agreement it demands. The check is there to
#: catch a KINK — a one-sided derivative differing from the differentiated one
#: by its own size — so the tolerance sits far above the second-order
#: truncation of the difference itself.
ANCHOR_PROBE_STEP = 1.0e-3
ANCHOR_TOLERANCE = 1.0e-4


class SeparatrixJumpError(NotImplementedError):
    """Raised when a continuation leaves a separatrix jump nothing owns."""


def separatrix_derivatives(gradient: Callable, orders: int) -> tuple[jax.Array, ...]:
    """Return a flux function and its derivatives at the separatrix.

    Differentiation is automatic, so the anchor is the declared callable's own
    derivative rather than a difference of it, and a continuation built from it
    stays differentiable in whatever device arrays the callable closes over.
    """
    separatrix = jnp.asarray(1.0, dtype=jnp.float64)
    derivative = gradient
    values = [jnp.asarray(gradient(separatrix))]
    for _ in range(orders - 1):
        derivative = jax.grad(derivative)
        values.append(jnp.asarray(derivative(separatrix)))
    return tuple(values)


def _reject_a_kinked_anchor(gradient: Callable, name: str, derivative) -> None:
    """Refuse a core gradient with no unambiguous separatrix derivative.

    A profile clipped or spliced at the separatrix carries one slope from
    inside the core and another at the point itself, so the continuity class
    would be met against a derivative the core never approaches. The one-sided
    second-order difference from inside is what a reader would measure, and it
    has to agree with the differentiated value.
    """
    inside = [
        float(jnp.asarray(gradient(jnp.asarray(1.0 - offset * ANCHOR_PROBE_STEP))))
        for offset in (0, 1, 2)
    ]
    one_sided = (3.0 * inside[0] - 4.0 * inside[1] + inside[2]) / (
        2.0 * ANCHOR_PROBE_STEP
    )
    analytic = float(jnp.asarray(derivative))
    scale = max(abs(one_sided), abs(analytic))
    if abs(analytic - one_sided) > ANCHOR_TOLERANCE * scale:
        raise ValueError(
            f"the declared {name} has no single derivative at the separatrix: "
            f"differentiating it gives {analytic:.6g} while approaching from "
            f"inside the core gives {one_sided:.6g}. A continuation anchored "
            "on a kink meets its continuity class against a slope the core "
            "never reaches"
        )


def _falling(power: int, order: int) -> float:
    """Return the coefficient the ``order``-th derivative gives ``d**power``."""
    if order > power:
        return 0.0
    return float(math.factorial(power) // math.factorial(power - order))


def _hermite_coefficients(targets, support: float) -> jax.Array:
    """Return the ascending coefficients of the minimal two-ended polynomial.

    The low half is fixed exactly by the separatrix targets, one Taylor
    coefficient each, so the declared continuity is met by construction rather
    than by a solve. The high half is what makes the taper and its matched
    derivatives vanish at the support bound, and it is a fixed linear map of
    the same targets — the anchor stays differentiable through it.
    """
    orders = len(targets)
    low = jnp.stack(
        [target / math.factorial(order) for order, target in enumerate(targets)]
    )
    condition = np.array(
        [
            [
                _falling(power, order) * support ** max(power - order, 0)
                for power in range(2 * orders)
            ]
            for order in range(orders)
        ]
    )
    outer = -np.linalg.solve(condition[:, orders:], condition[:, :orders])
    return jnp.concatenate([low, jnp.asarray(outer) @ low])


def _decay_coefficients(targets, decay_width: float) -> jax.Array:
    """Return the ascending polynomial prefactor of an exponential decay.

    Scaling the taper by :math:`e^{d/w}` turns it into its own prefactor, so
    the prefactor's Taylor coefficients are the Leibniz sum of the declared
    targets against the derivatives of the exponential.
    """
    return jnp.stack(
        [
            sum(
                math.comb(order, matched)
                * targets[matched]
                / decay_width ** (order - matched)
                for matched in range(order + 1)
            )
            / math.factorial(order)
            for order in range(len(targets))
        ]
    )


def _decay_integral_prefactor(prefactor: jax.Array, decay_width: float) -> jax.Array:
    """Return the prefactor of the exponential decay's own primitive.

    A primitive of :math:`Q(d) e^{-d/w}` has the form
    :math:`-w B(d) e^{-d/w}` with :math:`B - w B' = Q`, which is one downward
    recursion in the coefficients and exact.
    """
    orders = prefactor.size
    coefficients = [prefactor[orders - 1]]
    for order in range(orders - 2, -1, -1):
        coefficients.insert(
            0, prefactor[order] + decay_width * (order + 1) * coefficients[0]
        )
    return jnp.stack(coefficients)


@dataclass(frozen=True)
class _PolynomialTaper:
    """Two-ended polynomial taper of one gradient in separatrix distance."""

    coefficients: jax.Array
    support: float

    def __call__(self, distance: jax.Array) -> jax.Array:
        """Return the tapered gradient, exactly zero beyond the support."""
        value = jnp.polyval(self.coefficients[::-1], distance)
        return jnp.where(distance <= self.support, value, 0.0)

    def integral(self, distance: jax.Array) -> jax.Array:
        """Return the taper integrated from the separatrix out to a distance."""
        primitive = jnp.concatenate(
            [
                jnp.zeros(1, dtype=self.coefficients.dtype),
                self.coefficients / jnp.arange(1, self.coefficients.size + 1),
            ]
        )
        return jnp.polyval(primitive[::-1], jnp.minimum(distance, self.support))

    @property
    def edge_value(self) -> float:
        """Return the amplitude the taper still carries at the support bound.

        Read on the host: the bound is a declared constant, so the amplitude
        at it is a property of the declaration and must not become a traced
        computation inside a solve that reports it.
        """
        return float(np.polyval(np.asarray(self.coefficients)[::-1], self.support))


@dataclass(frozen=True)
class _DecayTaper:
    """Exponential taper of one gradient, truncated at the support bound."""

    prefactor: jax.Array
    integral_prefactor: jax.Array
    decay_width: float
    support: float

    def __call__(self, distance: jax.Array) -> jax.Array:
        """Return the tapered gradient, exactly zero beyond the support."""
        value = jnp.polyval(self.prefactor[::-1], distance) * jnp.exp(
            -distance / self.decay_width
        )
        return jnp.where(distance <= self.support, value, 0.0)

    def integral(self, distance: jax.Array) -> jax.Array:
        """Return the taper integrated from the separatrix out to a distance."""
        bounded = jnp.minimum(distance, self.support)
        primitive = (
            -self.decay_width
            * jnp.exp(-bounded / self.decay_width)
            * jnp.polyval(self.integral_prefactor[::-1], bounded)
        )
        return primitive + self.decay_width * self.integral_prefactor[0]

    @property
    def edge_value(self) -> float:
        """Return the amplitude the truncation at the support bound discards.

        Read on the host, for the same reason the polynomial family reads its
        own: the discarded step is a property of the declaration.
        """
        return float(
            np.polyval(np.asarray(self.prefactor)[::-1], self.support)
            * np.exp(-self.support / self.decay_width)
        )


@dataclass(frozen=True)
class SeparatrixContinuation:
    """Declared policy of one bounded continuation beyond the separatrix.

    ``support`` is the outer bound in separatrix distance, normalised flux
    measured away from the separatrix on the domain's own branch, and it is
    required: an unbounded continuation would drive current on every open cell
    the grid holds, out to the material boundary, with no declared physics
    saying it should.

    ``decay_width`` belongs to the exponential family alone and is refused on
    the polynomial one, so a width a form would ignore cannot sit in a source
    declaration looking as though it did something.
    """

    form: ContinuationForm
    continuity: SeparatrixContinuity
    support: float
    decay_width: float | None = None

    def __post_init__(self):
        """Validate the declared class, form and bounds."""
        if self.continuity is SeparatrixContinuity.UNDECLARED:
            raise ValueError(
                "a continuation must declare how many orders it matches at "
                "the separatrix; the admissible minimum is "
                f"{SeparatrixContinuity.VALUE_AND_GRADIENT.name.lower()}"
            )
        if self.continuity < SeparatrixContinuity.VALUE_AND_GRADIENT:
            missing = (
                "a toroidal surface current on the separatrix"
                if self.continuity is SeparatrixContinuity.VALUE_JUMP
                else "a thin current layer straddling the separatrix"
            )
            raise SeparatrixJumpError(
                f"{self.continuity.name.lower()} leaves a jump in the source "
                f"at the separatrix, which is carried by {missing}. No sheet, "
                "current-layer, sheath or halo model is declared in this "
                "solve and field-aligned scrape-off current waits on a "
                "specified material return path, so the jump has no owner; "
                "declare at least "
                f"{SeparatrixContinuity.VALUE_AND_GRADIENT.name.lower()}"
            )
        if self.form is ContinuationForm.UNDECLARED:
            raise ValueError("a continuation must declare its functional form")
        if not float(self.support) > 0.0:
            raise ValueError("support must be a positive separatrix distance")
        if self.form is ContinuationForm.EXPONENTIAL_DECAY:
            if self.decay_width is None or not float(self.decay_width) > 0.0:
                raise ValueError(
                    "an exponential continuation needs a positive decay width "
                    "in separatrix distance"
                )
        elif self.decay_width is not None:
            raise ValueError(
                f"the {self.form.name.lower()} family carries no decay width; "
                "its scale is the declared support"
            )

    def _taper(self, gradient: Callable, name: str, outward: float):
        """Return the taper of one core gradient on one branch."""
        orders = self.continuity.matched_orders
        derivatives = separatrix_derivatives(gradient, orders)
        _reject_a_kinked_anchor(gradient, name, derivatives[1])
        targets = tuple(
            outward**order * value for order, value in enumerate(derivatives)
        )
        if self.form is ContinuationForm.HERMITE_POLYNOMIAL:
            return derivatives, _PolynomialTaper(
                coefficients=_hermite_coefficients(targets, float(self.support)),
                support=float(self.support),
            )
        width = float(self.decay_width)
        prefactor = _decay_coefficients(targets, width)
        return derivatives, _DecayTaper(
            prefactor=prefactor,
            integral_prefactor=_decay_integral_prefactor(prefactor, width),
            decay_width=width,
            support=float(self.support),
        )

    def extend(
        self, inner: DomainProfile, domain: PlasmaDomain
    ) -> ContinuedDomainProfile:
        """Return the closure this policy continues one core profile into.

        The core profile is read once, at construction, for the value and
        derivatives the continuity class needs. Nothing of it is retained: the
        continuation is a closure of its own on its own domain, which is what
        lets the private-flux branch be varied without touching the common
        scrape-off layer.
        """
        if domain not in OUTWARD_SENSE:
            raise ValueError(
                f"{domain.name.lower()} is not an open domain; a continuation "
                "runs outward from the separatrix onto "
                f"{' or '.join(sorted(key.name.lower() for key in OUTWARD_SENSE))}"
            )
        if inner.rotation_closure is not RotationClosure.STATIC:
            raise NotImplementedError(
                f"a {inner.rotation_closure.name.lower()} closure makes the "
                "pressure source depend on major radius, so its separatrix "
                "anchor is not a flux function; continuing it needs the "
                "temperature and rotation primitives continued as well"
            )
        outward = OUTWARD_SENSE[domain]
        pressure_anchor, pressure = self._taper(inner.p_prime, "p_prime", outward)
        diamagnetic_anchor, diamagnetic = self._taper(
            inner.ff_prime, "ff_prime", outward
        )
        discarded = max(
            _relative_edge_amplitude(pressure, pressure_anchor),
            _relative_edge_amplitude(diamagnetic, diamagnetic_anchor),
        )
        return ContinuedDomainProfile(
            p_prime=_flux_function(pressure, outward),
            ff_prime=_flux_function(diamagnetic, outward),
            domain=domain,
            policy=self,
            outward=outward,
            pressure_anchor=pressure_anchor,
            diamagnetic_anchor=diamagnetic_anchor,
            pressure_taper=pressure,
            diamagnetic_taper=diamagnetic,
            truncated_fraction=discarded,
        )


def _relative_edge_amplitude(taper, anchor) -> float:
    """Return the amplitude at the support bound against the separatrix value.

    Zero when the separatrix value is zero: a taper anchored at nothing
    discards nothing, whatever its shape.
    """
    scale = abs(float(jnp.asarray(anchor[0])))
    if scale == 0.0:
        return 0.0
    return abs(taper.edge_value) / scale


def _separatrix_distance(psi_norm: jax.Array, outward: float) -> jax.Array:
    """Return the distance from the separatrix on one branch."""
    return jnp.maximum(outward * (jnp.asarray(psi_norm) - 1.0), 0.0)


def _flux_function(taper, outward: float) -> Callable:
    """Return one taper as a gradient of normalised flux."""

    def gradient(psi_norm):
        """Return the continued gradient at one normalised flux."""
        return taper(_separatrix_distance(psi_norm, outward))

    return gradient


@dataclass(frozen=True, kw_only=True)
class ContinuedDomainProfile(DomainProfile):
    """Core flux-function gradients continued onto one open domain.

    The gradients are ordinary flux functions of normalised flux, so the
    source evaluation, the current ledger and the conservation receipts read
    this closure through exactly the same seam they read the core through. What
    it adds is the receipt of how it was built and the two primitives its own
    branch owns: pressure and the toroidal-field function integrate outward
    from the boundary values, in closed form, and stay constant beyond the
    support bound where the gradients are zero.
    """

    domain: PlasmaDomain
    policy: SeparatrixContinuation
    outward: float
    pressure_anchor: tuple
    diamagnetic_anchor: tuple
    pressure_taper: object
    diamagnetic_taper: object
    truncated_fraction: float

    def __post_init__(self):
        """Validate the continued gradients and the declared domain."""
        super().__post_init__()
        if self.domain not in OUTWARD_SENSE:
            raise ValueError("a continuation is declared on an open domain")
        for name in ("pressure_taper", "diamagnetic_taper"):
            _validate_flux_function(getattr(self, name), name)

    def separatrix_distance(self, psi_norm: jax.Array) -> jax.Array:
        """Return the distance from the separatrix on this domain's branch."""
        return _separatrix_distance(psi_norm, self.outward)

    def _tail(self, taper, psi_norm: jax.Array) -> jax.Array:
        """Return the gradient integral in the core's boundary-inward sense.

        The core integrates from the evaluation point out to the boundary; on
        an open branch the same integral runs the other way from the
        separatrix, so one sign carries the whole difference and both
        primitives keep the form pinned in
        :mod:`nova.equilibrium.convention`.
        """
        return -self.outward * taper.integral(self.separatrix_distance(psi_norm))

    def pressure(
        self,
        radius: jax.Array,
        psi_norm: jax.Array,
        boundary_pressure,
        flux_span,
    ) -> jax.Array:
        """Return the pressure [Pa] the continuation carries on its domain."""
        return flux_function_pressure(
            boundary_pressure, flux_span, self._tail(self.pressure_taper, psi_norm)
        )

    def field_function_squared(
        self, psi_norm: jax.Array, boundary_field_function, flux_span
    ) -> jax.Array:
        """Return the squared toroidal-field function [T^2 m^2] on its domain."""
        return flux_function_toroidal_field(
            boundary_field_function,
            flux_span,
            self._tail(self.diamagnetic_taper, psi_norm),
        )

    def validate_separatrix_match(self, inner: DomainProfile) -> None:
        """Refuse a continuation anchored on anything but the declared core.

        A continuation built from a different profile — a lower private-flux
        pressure, say — meets its own continuity class perfectly and still
        leaves a jump at the separatrix of the solve it is used in. The
        independence a private-flux policy carries is in its form, width and
        support, never in the value or slope it starts from.
        """
        for name, anchor in (
            ("p_prime", self.pressure_anchor),
            ("ff_prime", self.diamagnetic_anchor),
        ):
            declared = separatrix_derivatives(getattr(inner, name), len(anchor))
            for order, (continued, core) in enumerate(
                zip(anchor, declared, strict=True)
            ):
                left, right = float(jnp.asarray(continued)), float(jnp.asarray(core))
                scale = max(abs(left), abs(right))
                if abs(left - right) > ANCHOR_TOLERANCE * scale:
                    raise SeparatrixJumpError(
                        f"the {self.domain.name.lower()} continuation starts "
                        f"from {name} derivative {order} = {left:.6g} while "
                        f"the core closure carries {right:.6g}; the "
                        "discontinuity that leaves at the separatrix is a "
                        "current sheet no model here owns"
                    )

    def continuation_record(self, dtype=jnp.float64) -> ContinuationRecord:
        """Return the receipt of this declared continuation."""
        declared = self.policy.decay_width
        width = jnp.nan if declared is None else float(declared)
        return ContinuationRecord(
            domain=jnp.asarray(int(self.domain), dtype=jnp.int8),
            form=jnp.asarray(int(self.policy.form), dtype=jnp.int8),
            continuity=jnp.asarray(int(self.policy.continuity), dtype=jnp.int8),
            support=jnp.asarray(float(self.policy.support), dtype=dtype),
            decay_width=jnp.asarray(width, dtype=dtype),
            separatrix_pressure_gradient=jnp.asarray(
                self.pressure_anchor[0], dtype=dtype
            ),
            separatrix_diamagnetic_gradient=jnp.asarray(
                self.diamagnetic_anchor[0], dtype=dtype
            ),
            truncated_fraction=jnp.asarray(self.truncated_fraction, dtype=dtype),
        )
