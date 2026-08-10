"""Source-free poloidal-flux basis on the ring functions of a focal circle.

Where no current flows the poloidal flux solves the HOMOGENEOUS Grad-Shafranov
operator exactly,

    Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2 = 0,

and that operator separates in the toroidal coordinates of a focal circle of
radius ``a`` at height ``z0``.  Writing ``eta = log(d_far/d_near)`` for the two
focal distances and ``theta`` for the angle about the circle, every solution is a
sum of

    psi_n = R sqrt(cosh eta - cos theta) F(cosh eta) {cos n theta, sin n theta}

with ``F`` an ORDER-ONE half-integer-degree Legendre function.  Order one, not
the scalar Laplace order zero, is what the ``-(1/R) dpsi/dR`` term asks for; the
explicit ``R`` is the total-flux convention (a column is a flux in Wb, so a flux
loop reads a column directly and a field probe reads its curl).

Two radial families, and which side each one's source is on
-----------------------------------------------------------
``eta`` grows toward the focal circle and vanishes on BOTH the symmetry axis and
at infinity, which are one place in these coordinates.  The two solutions of the
radial equation separate exactly on that:

* the FIRST kind ``P^1_{n-1/2}`` vanishes at ``eta = 0`` and grows toward the
  focal circle, so a sum of these is regular out to the axis and to infinity and
  its only singular set is the focal circle.  It represents whatever current sits
  BETWEEN THE OBSERVER AND THE FOCAL CIRCLE -- the enclosed side.
* the SECOND kind ``Q^1_{n-1/2}`` decays toward the focal circle and diverges at
  ``eta = 0``.  It is regular through the focal circle and represents current
  BEYOND THE OBSERVER -- external coils, the far field, anything nearer the axis
  or further out than the observation point.

A surface of constant ``eta`` is a circle in the poloidal plane, so the two
families are exactly the interior and exterior expansions of a shell, and a
measurement set spanning a shell that CONTAINS a conductor is reproducible by
neither.  That is the localisation lever: a filament at focal coordinates
``(eta_f, theta_f)`` enters the first-kind expansion with coefficients

    c_n = (4 mu0 I R_f / a) sqrt(cosh eta_f - cos theta_f)
          eps_n Q^1_{n-1/2}(cosh eta_f) / (1 - 4 n^2) {cos n theta_f, sin n theta_f}

(``eps_0 = 1``, ``eps_n = 2``), so the coefficient MODULUS falls at a rate set by
the source's distance from the focal circle and the coefficient PHASE advances at
``n`` times the source's angle about it.  Reading the two off a fitted expansion
recovers the source position without ever scanning candidate locations; the same
expression with the families exchanged holds for an observer on the other side.

Evaluating the ladders
----------------------
Both families climb the same three-term degree recurrence

    (n - 1/2) F_{n+1} = 2 n x F_n - (n + 1/2) F_{n-1},   x = cosh eta,

whose two solutions grow and decay like ``exp(+/- n eta)``.  The first kind is the
dominant one and is climbed forward from closed-form seeds.  The second kind is
the RECESSIVE one and forward climbing destroys it: measured against a quadrature
of the integral representation, a forward second-kind ladder is wrong by a factor
of ``3e12`` at degree eight once ``x`` reaches thirty -- a target a few centimetres
from a metre-scale focal circle.  It is therefore built by the ratio form of the
backward recurrence, seeded at zero far above the required degree and normalised
onto the exact lowest seed; the padding needed for a given ``x`` is set by the same
``exp(-2 n eta)`` separation that breaks the forward climb, so it is computed from
the geometry rather than guessed.

The seeds are complete elliptic integrals of the exponential coordinate
``t = exp(eta) = x + sqrt(x^2 - 1)``:

    P_{-1/2} = (2/pi) (2 sqrt(t)/(t+1)) K,  P_{1/2} = (2/pi) ((t+1)/sqrt(t) E - ...)
    Q_{-1/2} = 2 K / sqrt(t),               Q_{1/2} = 2 sqrt(t) (K - E)

each pair taken at its own modulus, and both moduli are handed to
:mod:`nova.biot.completeelliptic` as COMPLEMENTS formed from ``t`` without a
subtraction from one -- ``4t/(t+1)^2`` for the first kind and ``(t-1)(t+1)/t^2``
for the second.  The first kind's complement collapses toward the focal circle
and the second kind's toward the axis, which is exactly where each ladder's own
first kind diverges logarithmically, so a parameter would cap the accuracy at the
one place the caller cares about.  ``K - E`` is the one subtraction that survives:
it is the difference of two numbers approaching ``pi/2`` and loses about
``eps t^2`` relatively, which is a tenth of a digit a metre from the focal circle
and half the mantissa a micrometre from it.  Only ``Q_{1/2}`` uses it, and only
through ``Q^1_{-1/2}``, where it is added to a term larger by ``t``, so the loss
never reaches the ladder.

Radial domain
-------------
The order raise divides by ``sqrt(x^2 - 1)`` and the two independent radial
solutions coalesce as ``x = cosh(eta)`` approaches one.  Returning a value at a
held ``x`` would silently answer at a different point, and is especially wrong
for the second kind, which genuinely diverges there.  The public ladders instead
require ``x - 1 >= MINIMUM_COSH_GAP`` and reject the axis/infinity boundary.
The bound is tied to the backward recurrence's finite padding: at its boundary,
100-decimal references through degree eight give worst relative errors below
``4e-11`` for the first kind and ``4e-12`` for the second.  The coordinates may
still represent ``eta = 0`` exactly; only evaluation of a radial ladder there is
outside the numerical contract.

Conditioning
------------
The angular factors are orthogonal on a full turn, so an evaluation set that
samples the angle evenly gives a nearly orthogonal design for free; a real sensor
set does not, and the radial factors additionally span decades between the lowest
and highest degree.  :func:`solve_equilibrated` therefore scales every column to
unit norm ON THE FIT WINDOW before solving and truncates on the singular spectrum
of the SCALED design, so the discarded directions are the ones the measurement
cannot see rather than the ones whose natural units happen to be small.  Once the
rows are whitened by their noise, the same routine's significance cut is the
regularisation that matters: each singular direction's data projection then has
unit noise variance, so a threshold in standard deviations separates directions
the measurement determined from directions it did not, which no purely geometric
floor can do.  Degree is chosen by held-out prediction (:func:`select_order`),
never by inspecting the in-sample residual, which falls monotonically with degree
whether or not the added degrees carry information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize  # type: ignore[import-untyped]
import scipy.special  # type: ignore[import-untyped]

from nova.biot.completeelliptic import complete_kind

MU0 = 4.0e-7 * np.pi
"""Vacuum permeability [T.m/A]."""

INNER = "inner"
"""Radial family carrying current between the observer and the focal circle."""

OUTER = "outer"
"""Radial family carrying current beyond the observer."""

MINIMUM_COSH_GAP = 1.0e-5
"""Smallest supported ``cosh(eta) - 1`` for either radial ladder.

The boundary keeps the ratio recurrence within its finite backward padding and
is an accuracy domain, not a replacement value for points nearer ``eta = 0``.
"""

# Padding of the backward second-kind recurrence.  Its convergence is the same
# ``t^{-2}`` per step that makes the forward climb fail, so the trip count that
# brings the whole double range is ``log(1/eps)/(2 log t)``.  The floor covers
# well-separated solutions; the cap and ``MINIMUM_COSH_GAP`` jointly define the
# nearest supported coordinate, so the recurrence is never truncated and passed
# off as an accurate value.
_BACKWARD_PAD_FLOOR = 24
_BACKWARD_PAD_CAP = 4096

_MINIMUM_DISTANCE = float(np.arccosh(1.0 + MINIMUM_COSH_GAP))

# Floor on the product of focal distances, which vanishes only ON the focal
# circle.  The inner family genuinely diverges there, so the floor bounds a real
# infinity rather than answering for a nearby point: it returns a very large
# ``cosh eta`` instead of a non-number, which keeps a masked grid usable.
_SEPARATION_FLOOR = 1.0e-24


@dataclass(frozen=True)
class FocalCircle:
    """The circle the harmonics separate about."""

    radius: float
    """Focal-circle radius [m]."""

    height: float = 0.0
    """Focal-circle height [m]."""


@dataclass(frozen=True)
class FocalFrame:
    """Toroidal coordinates of a point set about a focal circle.

    ``cosine`` is ``cosh eta`` and ``gap`` is ``cosh eta - cos theta``, the
    combination the flux prefactor and the whole conformal Jacobian are built
    from.  ``radial_gradient`` and ``height_gradient`` hold the two independent
    entries of that Jacobian: the map from ``(R, Z)`` to ``(eta, -theta)`` is
    conformal, so a single complex derivative carries all four partials and
    ``deta/dR = -dtheta/dZ`` and ``deta/dZ = dtheta/dR`` by construction.
    """

    cosine: np.ndarray
    """``cosh eta`` at each point, including one on the axis/infinity boundary."""

    sine: np.ndarray
    """``sinh eta`` at each point."""

    angle: np.ndarray
    """``theta`` at each point, on ``(-pi, pi]``."""

    gap: np.ndarray
    """``cosh eta - cos theta``."""

    radial_gradient: np.ndarray
    """``deta/dR``, which is also ``-dtheta/dZ``."""

    height_gradient: np.ndarray
    """``deta/dZ``, which is also ``dtheta/dR``."""

    @property
    def distance(self) -> np.ndarray:
        """Return ``eta``, the logarithmic distance from the focal circle."""
        return np.arccosh(self.cosine)


def focal_frame(r, z, focus: FocalCircle) -> FocalFrame:
    """Return the toroidal frame of ``(r, z)`` about ``focus``.

    The forward map is ``R = a sinh eta / gap`` and ``Z = z0 + a sin theta / gap``
    with ``gap = cosh eta - cos theta``, whose inverse follows from the two focal
    distances ``d_near`` to ``(a, z0)`` and ``d_far`` to ``(-a, z0)``:

        cosh eta = (R^2 + dz^2 + a^2) / (d_near d_far),
        cos theta = (R^2 + dz^2 - a^2) / (d_near d_far),
        sin theta = 2 a dz / (d_near d_far),
        gap = 2 a^2 / (d_near d_far).

    Every one of them is a ratio of quantities the caller's geometry gives
    directly, so ``gap`` -- which the flux prefactor takes a square root of --
    never arrives as a difference of two large numbers.
    """
    radius = float(focus.radius)
    r = np.asarray(r, dtype=np.float64)
    height = np.asarray(z, dtype=np.float64) - float(focus.height)
    near = (r - radius) ** 2 + height**2
    far = (r + radius) ** 2 + height**2
    separation = np.sqrt(np.maximum(near * far, _SEPARATION_FLOOR))
    square = r**2 + height**2
    total = square + radius**2
    # Factor the small coordinate distance instead of subtracting one from
    # total / separation.  The numerator follows from
    # total^2 - near*far = 4 a^2 R^2, so it is positive and retains the public
    # radial boundary through a position -> frame round trip.
    cosine_gap = 4.0 * radius**2 * r**2 / (separation * (total + separation))
    cosine = 1.0 + cosine_gap
    gap = 2.0 * radius**2 / separation
    angle = np.arctan2(
        2.0 * radius * height / separation, (square - radius**2) / separation
    )
    # The conformal derivative of eta - i theta with respect to R + i dz is
    # -2 a / ((R + i dz)^2 - a^2); its real and imaginary parts are the two
    # independent Jacobian entries, and the squared modulus of the denominator is
    # the squared focal-distance product, which is 4 a^4 / gap^2.
    real = r**2 - height**2 - radius**2
    imaginary = 2.0 * r * height
    scale = gap**2 / (2.0 * radius**3)
    return FocalFrame(
        cosine=cosine,
        sine=np.sqrt(cosine_gap * (2.0 + cosine_gap)),
        angle=angle,
        gap=gap,
        radial_gradient=-real * scale,
        height_gradient=-imaginary * scale,
    )


def focal_position(
    focus: FocalCircle, distance, angle
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, Z)`` from the focal coordinates ``(eta, theta)``.

    The denominator is formed from ``v = 2 tanh(eta/2)`` and the angular
    half-angle as ``(2 sin(theta/2))^2 + cos(theta/2)^2 v^2``.  Both terms are
    divided by their largest magnitude before squaring, so coordinates whose
    squares would underflow remain resolved.  Every intermediate stays bounded
    for finite ``eta``.  The height numerator uses ``sech(eta/2)^2`` formed from
    ``exp(-abs(eta))``, retaining the representable tail as the point approaches
    the focal circle.  Exactly ``eta = theta = 0`` is the coordinate point at
    infinity and is path dependent, so it raises instead of inventing finite
    coordinates.
    """
    distance, angle = np.broadcast_arrays(
        np.asarray(distance, dtype=np.float64),
        np.asarray(angle, dtype=np.float64),
    )
    half_angle = 0.5 * angle
    sine = np.sin(half_angle)
    cosine = np.cos(half_angle)
    exponential = np.exp(-np.abs(distance))
    sech = 2.0 * exponential / (1.0 + exponential**2)
    twice_tangent = 2.0 * np.tanh(distance) / (1.0 + sech)
    twice_sine = 2.0 * sine
    twice_sine = np.where((twice_sine == 0.0) & (angle != 0.0), angle, twice_sine)
    scale = np.maximum(np.abs(twice_tangent), np.abs(twice_sine))
    if np.any(scale == 0.0):
        raise ValueError("eta = theta = 0 is the path-dependent point at infinity")
    normalized_tangent = twice_tangent / scale
    normalized_sine = twice_sine / scale
    denominator = normalized_sine**2 + cosine**2 * normalized_tangent**2
    squared_sech = 4.0 * exponential / (1.0 + exponential) ** 2
    return (
        2.0 * focus.radius * normalized_tangent / denominator / scale,
        focus.height
        + 2.0
        * focus.radius
        * normalized_sine
        * cosine
        * squared_sech
        / denominator
        / scale,
    )


# --- the two radial ladders -------------------------------------------------


def _radial_argument(order: int, x) -> np.ndarray:
    """Return a validated radial argument inside the public accuracy domain."""
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    x = np.asarray(x, dtype=np.float64)
    minimum = 1.0 + MINIMUM_COSH_GAP
    if np.any(~np.isfinite(x)) or np.any(x < minimum):
        raise ValueError(
            f"radial ladders require finite x with x - 1 >= {MINIMUM_COSH_GAP:g}"
        )
    return x


def _seeds(x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return ``(sinh eta, exp eta)`` and three order-one lowest seeds.

    The order raise is ``F^1_nu = nu (x F_nu - F_{nu-1}) / sqrt(x^2 - 1)`` with
    the degree reflection ``F_{-3/2} = F_{1/2}``, which holds for both kinds --
    the second kind's reflection carries a cotangent that vanishes at this
    half-integer degree.  The first kind returns both reflected neighbours from
    closed forms.  The second kind returns only its stable lowest value; its
    reflected neighbour is obtained from the backward ratio chain, avoiding the
    cancellation in the raised ``K - E`` expression at large ``x``.
    """
    sine = np.sqrt((x - 1.0) * (x + 1.0))
    exponential = x + sine
    first_k, first_e = complete_kind(4.0 * exponential / (exponential + 1.0) ** 2)
    root = np.sqrt(exponential)
    low = 2.0 * root / (exponential + 1.0)
    first_low = 2.0 / np.pi * low * first_k
    first_high = 2.0 / np.pi * ((exponential + 1.0) / root * first_e - low * first_k)
    second_k, second_e = complete_kind(
        (exponential - 1.0) * (exponential + 1.0) / exponential**2
    )
    second_low = 2.0 * second_k / root
    second_high = 2.0 * root * (second_k - second_e)
    return (
        sine,
        exponential,
        -0.5 * (x * first_low - first_high) / sine,
        0.5 * (x * first_high - first_low) / sine,
        -0.5 * (x * second_low - second_high) / sine,
    )


def _degree_gradient(
    order: int, x: np.ndarray, ladder: np.ndarray, reflected: np.ndarray
) -> np.ndarray:
    """Return ``dF^1_{n-1/2}/dx`` for a complete order-one ladder.

    ``(x^2 - 1) dF^1_nu/dx = nu x F^1_nu - (nu + 1) F^1_{nu-1}`` at order one,
    with the same degree reflection the seeds use.
    """
    span = (x - 1.0) * (x + 1.0)
    out = np.empty_like(ladder)
    for n in range(order + 1):
        previous = reflected if n == 0 else ladder[n - 1]
        out[n] = ((n - 0.5) * x * ladder[n] - (n + 0.5) * previous) / span
    return out


def _first_kind_series(order: int, x: np.ndarray) -> np.ndarray:
    """Return first-kind order-one values from the regular-point series.

    Differentiating ``P_nu(x) = 2F1(-nu, nu + 1; 1; (1 - x) / 2)`` gives a
    direct expression whose terms are all resolved near ``x = 1``.  In that
    region it avoids deriving the smallest order-one values by subtracting the
    elliptic seeds.
    """
    span = np.sqrt((x - 1.0) * (x + 1.0))
    argument = 0.5 * (1.0 - x)
    values = []
    for n in range(order + 1):
        degree = n - 0.5
        values.append(
            0.5
            * degree
            * (degree + 1.0)
            * span
            * scipy.special.hyp2f1(1.0 - degree, degree + 2.0, 2.0, argument)
        )
    return np.stack(values)


def ring_legendre_first(order: int, x) -> tuple[np.ndarray, np.ndarray]:
    """Return ``P^1_{n-1/2}(x)`` and its ``x`` derivative for ``n = 0..order``.

    Near the regular point ``x = 1`` the hypergeometric series is evaluated
    directly, avoiding cancellation in the elliptic seed raise.  Farther away
    the dominant solution is climbed forward from the two elliptic-integral
    seeds.  The leading dimension of both returns is ``order + 1`` and the
    remaining dimensions are the broadcast shape of ``x``.

    ``x - 1`` must be at least :data:`MINIMUM_COSH_GAP`; the axis/infinity
    boundary is rejected rather than evaluated at a surrogate point.
    """
    x = _radial_argument(order, x)
    _sine, _exponential, low, high, _second_low = _seeds(x)
    ladder = [low, high]
    for n in range(1, order):
        ladder.append((2.0 * n * x * ladder[n] - (n + 0.5) * ladder[n - 1]) / (n - 0.5))
    stacked = np.stack(ladder[: order + 1])
    reflected = high
    regular = x <= 2.0
    if np.any(regular):
        series = _first_kind_series(max(order, 1), x)
        stacked = np.where(regular, series[: order + 1], stacked)
        reflected = np.where(regular, series[1], reflected)
    return stacked, _degree_gradient(order, x, stacked, reflected)


def ring_legendre_second(order: int, x) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Q^1_{n-1/2}(x)`` and its ``x`` derivative for ``n = 0..order``.

    The recessive solution: the recurrence is run DOWNWARD in ratio form from a
    vanishing seed far above the requested degree, which converges onto the
    recessive ratio at the same rate the forward climb diverges, and the resulting
    ratio chain is walked up from the exact lowest seed.  The ratio form neither
    overflows nor needs rescaling, so one padding serves the whole point set.

    The ratio chain is always carried through degree one, even for an order-zero
    request, because the reflected adjacent value is required by the derivative.
    The leading dimension of both returns is ``order + 1`` and the remaining
    dimensions are the broadcast shape of ``x``.

    ``x - 1`` must be at least :data:`MINIMUM_COSH_GAP`; the divergent
    axis/infinity boundary is rejected rather than clipped.
    """
    x = _radial_argument(order, x)
    _sine, exponential, _first_low, _first_high, low = _seeds(x)
    required = max(order, 1)
    separation = np.log(exponential.min(initial=np.inf))
    padding = int(
        np.clip(
            np.ceil(-np.log(np.finfo(np.float64).eps) / (2.0 * separation)),
            _BACKWARD_PAD_FLOOR,
            _BACKWARD_PAD_CAP,
        )
    )
    ratio = np.zeros_like(x)
    chain: dict[int, np.ndarray] = {}
    for n in range(required + padding, 0, -1):
        ratio = (n + 0.5) / (2.0 * n * x - (n - 0.5) * ratio)
        if n <= required:
            chain[n - 1] = ratio
    ladder = [low]
    for n in range(required):
        ladder.append(ladder[-1] * chain[n])
    stacked = np.stack(ladder[: order + 1])
    return stacked, _degree_gradient(order, x, stacked, ladder[1])


_LADDER = {INNER: ring_legendre_first, OUTER: ring_legendre_second}


# --- the basis --------------------------------------------------------------


@dataclass(frozen=True)
class ToroidalHarmonics:
    """A source-free flux basis about a focal circle.

    ``families`` selects which radial sides are carried: the inner family alone
    for a measurement set that encloses every source, the outer family alone for
    one that excludes them, and both to fit a shell with current on either side.
    """

    focus: FocalCircle
    order: int = 6
    families: tuple[str, ...] = (INNER,)

    def __post_init__(self):
        """Reject a family name the ladder table does not know."""
        unknown = [name for name in self.families if name not in _LADDER]
        if unknown:
            raise ValueError(
                f"unknown radial families {unknown}; expected {list(_LADDER)}"
            )
        if self.order < 0:
            raise ValueError(f"order must be non-negative, got {self.order}")

    @property
    def labels(self) -> list[str]:
        """Return the column labels, family by family then degree by degree."""
        out = []
        for family in self.families:
            for n in range(self.order + 1):
                out.append(f"{family}{n}" if n == 0 else f"{family}{n}c")
                if n >= 1:
                    out.append(f"{family}{n}s")
        return out

    def _angular(self, angle: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return each column's angular factor and its ``theta`` derivative."""
        out = []
        for n in range(self.order + 1):
            out.append((np.cos(n * angle), -n * np.sin(n * angle)))
            if n >= 1:
                out.append((np.sin(n * angle), n * np.cos(n * angle)))
        return out

    def _degree_index(self) -> np.ndarray:
        """Return the degree ``n`` of each column within one family."""
        degree = [0]
        for n in range(1, self.order + 1):
            degree += [n, n]
        return np.asarray(degree, dtype=int)

    def flux(self, r, z) -> np.ndarray:
        """Return the flux columns ``(n_point, n_column)`` [Wb per coefficient].

        Column ``k`` is a poloidal flux with ``Delta* psi = 0`` everywhere except
        its own family's singular set -- the focal circle for the inner family,
        the axis and infinity for the outer one.
        """
        r = np.asarray(r, dtype=np.float64)
        frame = focal_frame(r, z, self.focus)
        prefactor = r * np.sqrt(frame.gap)
        angular = self._angular(frame.angle)
        degree = self._degree_index()
        columns = []
        for family in self.families:
            radial = _LADDER[family](self.order, frame.cosine)[0]
            for k, (shape, _) in enumerate(angular):
                columns.append(prefactor * radial[degree[k]] * shape)
        return np.stack(columns, axis=1)

    def field(self, r, z) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(B_R, B_Z)`` columns of every flux column [T].

        Analytic throughout: the flux is differentiated in the focal coordinates,
        where the radial factor's derivative comes off the same recurrence that
        built it, and mapped to ``(R, Z)`` by the conformal Jacobian.  Under the
        total-flux convention ``B_R = -(1/2 pi R) dpsi/dZ`` and
        ``B_Z = +(1/2 pi R) dpsi/dR``.
        """
        r = np.asarray(r, dtype=np.float64)
        frame = focal_frame(r, z, self.focus)
        angular = self._angular(frame.angle)
        degree = self._degree_index()
        sine_theta = np.sin(frame.angle)
        # cosh(eta) * gap - sinh(eta)^2 / 2 is this positive sum.
        # The identity retains the small distance derivative when both terms in
        # the direct difference agree to nearly every floating-point digit.
        distance_numerator = 0.5 * (frame.gap**2 + sine_theta**2)
        radial_parts, height_parts = [], []
        for family in self.families:
            radial, gradient = _LADDER[family](self.order, frame.cosine)
            for k, (shape, slope) in enumerate(angular):
                value = radial[degree[k]]
                derivative = frame.sine * gradient[degree[k]]
                # d(psi)/d(eta) and d(psi)/d(theta) of R sqrt(gap) F(cosh eta) g(theta)
                by_distance = (
                    self.focus.radius
                    / frame.gap**1.5
                    * shape
                    * (distance_numerator * value + frame.sine * frame.gap * derivative)
                )
                by_angle = (
                    self.focus.radius
                    * frame.sine
                    / frame.gap**1.5
                    * value
                    * (frame.gap * slope - 0.5 * sine_theta * shape)
                )
                radial_parts.append(
                    by_distance * frame.radial_gradient
                    + by_angle * frame.height_gradient
                )
                height_parts.append(
                    by_distance * frame.height_gradient
                    - by_angle * frame.radial_gradient
                )
        circumference = 2.0 * np.pi * r
        by_radius = np.stack(radial_parts, axis=1) / circumference[:, None]
        by_height = np.stack(height_parts, axis=1) / circumference[:, None]
        return -by_height, by_radius

    def project(self, r, z, cosine, sine) -> np.ndarray:
        """Return field columns projected onto per-point sensitive axes.

        ``cosine`` and ``sine`` are the radial and vertical components of each
        point's unit sensitive direction, the pose convention a pickup coil is
        described by.
        """
        radial, height = self.field(r, z)
        return (
            np.asarray(cosine, dtype=np.float64)[:, None] * radial
            + np.asarray(sine, dtype=np.float64)[:, None] * height
        )


def grad_shafranov_residual(psi, grid_r, grid_z) -> np.ndarray:
    """Return ``Delta* psi`` on an ``(n_z, n_r)`` raster, edges as NaN.

    Second-order central differences on the uniform grid.  This is the basis's
    correctness oracle, independent of where any analytic formula came from:
    every column must drive it to the truncation floor away from its own family's
    singular set.
    """
    psi = np.asarray(psi, dtype=np.float64)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    step_r = float(r[1] - r[0])
    step_z = float(z[1] - z[0])
    out = np.full_like(psi, np.nan)
    second_r = (psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / step_r**2
    first_r = (psi[:, 2:] - psi[:, :-2]) / (2.0 * step_r)
    second_z = (psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / step_z**2
    out[1:-1, 1:-1] = (
        second_r[1:-1, :] - first_r[1:-1, :] / r[1:-1][None, :] + second_z[:, 1:-1]
    )
    return out


# --- the filament expansion -------------------------------------------------


def filament_coefficients(
    basis: ToroidalHarmonics, source_r: float, source_z: float, *, current: float = 1.0
) -> np.ndarray:
    """Return the exact expansion of a circular filament in a one-family basis.

    The filament sits at ``(source_r, source_z)`` and carries ``current`` amperes.
    For the inner family the expansion holds at every point OUTSIDE the filament's
    own focal circle (``eta < eta_f``, farther from the focal circle than the
    filament); for the outer family it holds inside.  Feeding the returned
    coefficients back through :meth:`ToroidalHarmonics.flux` reproduces
    :func:`nova.biot.greens.greens_psi` on the valid side.

    The two ladders exchange roles between the coefficient and the column: an
    inner-family column carries the first kind at the observer and the coefficient
    carries the SECOND kind at the source, which is what makes the coefficient
    modulus a direct readout of how far the source sits from the focal circle.
    """
    if len(basis.families) != 1:
        raise ValueError(
            f"a filament expansion is single-sided; basis carries {basis.families}"
        )
    family = basis.families[0]
    frame = focal_frame(
        np.array([float(source_r)]), np.array([float(source_z)]), basis.focus
    )
    opposite = OUTER if family == INNER else INNER
    radial = _LADDER[opposite](basis.order, frame.cosine)[0][:, 0]
    degree = np.arange(basis.order + 1)
    fold = np.where(degree == 0, 1.0, 2.0)
    weight = (
        4.0
        * MU0
        * float(current)
        * float(source_r)
        * float(np.sqrt(frame.gap[0]))
        / basis.focus.radius
        * fold
        * radial
        / (1.0 - 4.0 * degree**2)
    )
    angle = float(frame.angle[0])
    out = []
    for n in range(basis.order + 1):
        out.append(weight[n] * np.cos(n * angle))
        if n >= 1:
            out.append(weight[n] * np.sin(n * angle))
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class SourceEstimate:
    """A filament position and current read off a fitted one-family expansion."""

    r: float
    """Recovered filament radius [m]."""

    z: float
    """Recovered filament height [m]."""

    current: float
    """Recovered filament current [A]."""

    distance: float
    """Recovered ``eta_f``, the source's logarithmic distance from the focus."""

    angle: float
    """Recovered ``theta_f``, the source's angle about the focus."""

    modulus_residual: float
    """Fraction of the coefficient-modulus vector the single-source law misses.

    Zero for one filament and bounded by one; it grows with the number of
    independent sources and is the practical signal that a single-filament
    reading is inadequate.
    """

    phase_residual: float
    """Root-mean-square scatter of the coefficient phases about ``n theta_f`` [rad]."""


def locate_source(
    basis: ToroidalHarmonics,
    coefficients,
    *,
    degrees: int | None = None,
    current_sign: float | None = None,
) -> SourceEstimate:
    """Return the filament a one-family expansion's coefficients imply.

    The modulus of the degree-``n`` coefficient pair carries the source's distance
    through the opposite radial ladder divided by ``1 - 4 n^2``, whose ratios
    between consecutive degrees vary monotonically with distance.  The signed
    radial factors are removed before reading phase, so both radial families and
    either current direction leave ``n`` times the source's angle.  Distance is
    solved by a bounded search matching the modulus sequence with the amplitude
    profiled out, so the answer never depends on an absolute calibration; angle
    comes from the lowest degree, which is the unambiguous one, and the higher
    degrees are unwrapped against it and averaged with the modulus as weight.

    ``degrees`` truncates the sequence used, which is how a held-out degree
    selection is carried into the read.  ``current_sign`` may supply a known
    current direction when the coefficients are only an approximate single-source
    fit; otherwise the degree-zero coefficient determines it exactly.
    """
    if len(basis.families) != 1:
        raise ValueError(
            f"a source read is single-sided; basis carries {basis.families}"
        )
    if current_sign is not None and (
        not np.isfinite(current_sign) or current_sign == 0.0
    ):
        raise ValueError("current_sign must be finite and non-zero when supplied")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    top = basis.order if degrees is None else min(int(degrees), basis.order)
    if top < 1:
        raise ValueError("at least one non-zero degree is needed to place a source")
    cosine = np.empty(top + 1)
    sine = np.zeros(top + 1)
    cosine[0] = coefficients[0]
    for n in range(1, top + 1):
        cosine[n] = coefficients[2 * n - 1]
        sine[n] = coefficients[2 * n]
    modulus = np.hypot(cosine, sine)
    if modulus[0] <= 0.0 or np.count_nonzero(modulus) < 2:
        raise ValueError("coefficient sequence carries no resolvable source")

    degree = np.arange(top + 1)
    fold = np.where(degree == 0, 1.0, 2.0)
    shape = np.abs(fold / (1.0 - 4.0 * degree**2))
    opposite = OUTER if basis.families[0] == INNER else INNER

    def scatter(distance: float) -> float:
        """Return the amplitude-profiled modulus misfit at a trial distance.

        Matched in the coefficients' own linear units rather than in logarithms:
        the sequence spans decades, so a logarithmic match gives a degree sitting
        at the round-off floor the same weight as the degree that carries the
        source, and a truncated fit always has such degrees at the top.  Profiling
        the amplitude out leaves the direction cosine between the measured and
        modelled modulus vectors, which is scale-free and dominated by the
        degrees that were actually measured.
        """
        model = shape * np.abs(
            _LADDER[opposite](top, np.array([np.cosh(distance)]))[0][:, 0]
        )
        model_scale = float(np.max(model, initial=0.0))
        modulus_scale = float(np.max(modulus, initial=0.0))
        if not model_scale > 0.0 or not modulus_scale > 0.0:
            return 1.0
        normalized_model = model / model_scale
        normalized_modulus = modulus / modulus_scale
        amplitude = float(normalized_modulus @ normalized_model) / float(
            normalized_model @ normalized_model
        )
        residual = normalized_modulus - amplitude * normalized_model
        return float((residual @ residual) / (normalized_modulus @ normalized_modulus))

    search = scipy.optimize.minimize_scalar(
        scatter,
        bounds=(_MINIMUM_DISTANCE, 12.0),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    # The bounded solver samples only the open interval.  Compare its result to
    # the inclusive public boundary explicitly so a source exactly on that
    # boundary is not displaced inward by the optimiser's stopping distance.
    boundary_scatter = scatter(_MINIMUM_DISTANCE)
    distance = (
        _MINIMUM_DISTANCE if boundary_scatter <= float(search.fun) else float(search.x)
    )

    radial_shape = _LADDER[opposite](top, np.array([np.cosh(distance)]))[0][:, 0]
    signed_shape = fold * radial_shape / (1.0 - 4.0 * degree**2)
    weight = modulus[1:]
    harmonic_order = np.arange(1, top + 1)

    def phase_estimate(common_sign: float) -> tuple[float, float]:
        """Return angle and phase scatter for one possible current direction."""
        orientation = common_sign * np.sign(signed_shape)
        phase = np.arctan2(sine[1:] * orientation[1:], cosine[1:] * orientation[1:])
        lowest = phase[0]
        turns = np.round((harmonic_order * lowest - phase) / (2.0 * np.pi))
        unwrapped = (phase + 2.0 * np.pi * turns) / harmonic_order
        angle = float(
            np.arctan2(
                np.sum(weight * np.sin(unwrapped)),
                np.sum(weight * np.cos(unwrapped)),
            )
        )
        residual = float(
            np.sqrt(
                np.sum(weight * (np.angle(np.exp(1j * (unwrapped - angle)))) ** 2)
                / max(np.sum(weight), np.finfo(float).tiny)
            )
        )
        return angle, residual

    common_sign = (
        float(np.sign(cosine[0] / signed_shape[0]))
        if current_sign is None
        else float(np.sign(current_sign))
    )
    angle, phase_residual = phase_estimate(common_sign)

    source_r, source_z = focal_position(basis.focus, distance, angle)
    source_r = float(source_r)
    source_z = float(source_z)
    unit = filament_coefficients(basis, source_r, source_z, current=1.0)
    current = float(coefficients[0] / unit[0]) if unit[0] != 0.0 else float("nan")
    return SourceEstimate(
        r=source_r,
        z=source_z,
        current=current,
        distance=distance,
        angle=angle,
        modulus_residual=float(np.sqrt(scatter(distance))),
        phase_residual=phase_residual,
    )


def convergent_points(basis: ToroidalHarmonics, r, z, distance: float) -> np.ndarray:
    """Return the mask of points a one-family expansion is valid at.

    An inner-family expansion of a source at focal distance ``distance`` converges
    strictly OUTSIDE the source's own focal circle, and an outer-family expansion
    strictly inside.  Points on the wrong side are not merely inaccurate: the
    series diverges there, so they must be dropped from a fit rather than
    down-weighted.
    """
    frame = focal_frame(r, z, basis.focus)
    if basis.families[0] == INNER:
        return frame.distance < distance
    return frame.distance > distance


# --- conditioning and degree selection --------------------------------------


@dataclass(frozen=True)
class ColumnFit:
    """One equilibrated least-squares solve of a harmonic design."""

    coefficients: np.ndarray
    labels: list[str] = field(default_factory=list)
    singular_values: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    rank: int = 0
    raw_condition: float = float("nan")
    """Condition number of the design in its natural units."""

    equilibrated_condition: float = float("nan")
    """Condition number after every column is scaled to unit norm on the window."""

    residual: float = float("nan")
    """Root-mean-square in-sample residual, in the units of the weighted data.

    With a whitening weight that is a multiple of the noise, so one is the value
    a fit at the noise level returns.
    """


def _condition(values: np.ndarray) -> float:
    """Return a singular spectrum's condition number, infinite when singular."""
    if values.size == 0 or values.min() <= 0.0:
        return float("inf")
    return float(values.max() / values.min())


def solve_equilibrated(
    design,
    data,
    *,
    weight=None,
    floor: float = 1.0e-10,
    significance: float = 0.0,
) -> ColumnFit:
    """Solve a harmonic design by column-equilibrated truncated least squares.

    Columns are scaled to unit norm on the fit window before the solve, so the
    singular spectrum the truncation acts on measures what the measurement can
    RESOLVE rather than what the basis functions happen to be worth in their own
    units -- a degree-eight radial factor is smaller than a degree-zero one by
    decades at any realistic standoff, and truncating the unscaled spectrum throws
    away the high degrees before it has asked whether they were measured.
    ``floor`` is the relative singular-value cut on the scaled design.

    ``weight`` is a per-row multiplier applied to both sides, so passing the
    reciprocal of each row's noise gives the whitened solve.  In that whitened
    frame the projection of the data onto each left singular vector has unit
    noise variance whatever the design looks like, so ``significance`` is a
    threshold in standard deviations below which a singular direction is
    discarded as unmeasured.  This is the cut that matters once data is noisy: a
    direction whose projection is a fraction of a sigma contributes nothing but
    amplified noise, and no purely geometric floor can know where that line is.
    """
    design = np.asarray(design, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if weight is not None:
        weight = np.asarray(weight, dtype=np.float64)
        design = design * weight[:, None]
        data = data * weight
    norm = np.linalg.norm(design, axis=0)
    live = norm > 0.0
    scale = np.where(live, norm, 1.0)
    scaled = design / scale
    left, spectrum, right = np.linalg.svd(scaled, full_matrices=False)
    projection = left.T @ data
    keep = spectrum > floor * spectrum.max(initial=0.0)
    if significance > 0.0:
        keep &= np.abs(projection) > significance
    inverse = np.where(keep, 1.0 / np.where(keep, spectrum, 1.0), 0.0)
    coefficients = np.where(live, (right.T * inverse) @ projection / scale, 0.0)
    return ColumnFit(
        coefficients=coefficients,
        singular_values=spectrum,
        rank=int(keep.sum()),
        raw_condition=_condition(np.linalg.svd(design, compute_uv=False)),
        equilibrated_condition=_condition(spectrum),
        residual=float(np.sqrt(np.mean((design @ coefficients - data) ** 2))),
    )


def select_order(
    build,
    data,
    orders,
    *,
    folds: int = 5,
    weight=None,
    seed: int = 0,
    significance: float = 0.0,
):
    """Return the held-out-best degree and the score of every degree tried.

    ``build`` maps a degree to that degree's design matrix over the full row set.
    Rows are partitioned into ``folds`` interleaved groups after a fixed shuffle,
    each group is predicted from a fit to the others, and the score is the
    root-mean-square held-out prediction error pooled over folds.  In-sample
    residual is never used: it falls with degree whether or not the added degrees
    carry information, so choosing on it selects the highest degree offered every
    time.
    """
    data = np.asarray(data, dtype=np.float64)
    index = np.arange(data.size)
    np.random.default_rng(seed).shuffle(index)
    groups = [index[k::folds] for k in range(folds)]
    scores = {}
    for order in orders:
        design = np.asarray(build(order), dtype=np.float64)
        squared, count = 0.0, 0
        for held in groups:
            train = np.setdiff1d(index, held)
            row_weight = None if weight is None else np.asarray(weight)[train]
            fit = solve_equilibrated(
                design[train], data[train], weight=row_weight, significance=significance
            )
            squared += float(
                np.sum((design[held] @ fit.coefficients - data[held]) ** 2)
            )
            count += held.size
        scores[int(order)] = float(np.sqrt(squared / max(count, 1)))
    best = min(scores, key=lambda key: scores[key])
    return best, scores


__all__ = [
    "INNER",
    "MINIMUM_COSH_GAP",
    "OUTER",
    "ColumnFit",
    "FocalCircle",
    "FocalFrame",
    "SourceEstimate",
    "ToroidalHarmonics",
    "convergent_points",
    "filament_coefficients",
    "focal_frame",
    "focal_position",
    "grad_shafranov_residual",
    "locate_source",
    "ring_legendre_first",
    "ring_legendre_second",
    "select_order",
    "solve_equilibrated",
]
