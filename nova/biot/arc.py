"""Biot-Savart calculation for arc segments.

The finite arc stops the ring's range short, so every elliptic integral the
reduction takes regains an amplitude and none of them is complete.  What does NOT
change is the argument each one is handed: :mod:`nova.biot.constants` takes every
ring pole and the modulus as an exact geometric complement, and the rows here take
them the same way.  Concretely, four quantities are never formed by subtracting
from one:

    dn^2 = 1 - k^2 sin^2 theta  is  cos^2 theta + k'^2 sin^2 theta
    1 - n sin^2 theta           is  cos^2 theta + (1 - n) sin^2 theta
    k^2 - n                     is  (1 - n) - k'^2
    2 r - b k^2                 is  2 r (gamma^2 - b (rs - r))/a^2

and each right-hand side is a sum of positives, or a difference of two quantities
that are themselves of order the squared distance to a source corner.  All four
vanish at a confluence the reduction reaches routinely -- a target on the arc's own
END PLANE, where the amplitude is a quarter turn and the sine's square is one, or a
target on the source ring, where the modulus complement is the distance -- and the
printed forms return between four decades and no digits at all there.

The two Jacobian functions the rows carry, ``sn`` and ``cn``, are the amplitude's
own sine and cosine: inverting the first kind to recover them is a numerical round
trip with an exact answer, and once the parameter rounds to one it returns ``nan``
instead.  The first and second kinds come off the complement-native descent
(:mod:`nova.biot.incompleteelliptic`) rather than the parameter's entry points,
because the first grows like ``-log k'`` and an ``eps`` on the parameter is a
relative ``eps/k'^2`` on the answer whichever way the parameter is formed.  The
third kind takes the pole and the complement, which routes it through the descent
whose every term is positive; see :meth:`nova.biot.constants.Constants.ellippinc`
for why that is a domain split rather than a fallback.

One loss is NOT closed here and is measured rather than hidden.  The amplitude is
assembled as ``alpha = (pi - psi)/2`` for an azimuthal separation ``psi`` from an
arc end, so the separation's own digits are spent before any row sees it, and the
end rows' ``(sine, cosine)`` pair can only be as good as the assembled angle --
which leaves the first kind at 2.5e-12 for a target within 1e-08 radians of the
end plane AND within 1e-08 radii of the ring.  The two LIMIT rows are exempt,
because a vanishing amplitude and a right angle are exact by construction, and the
quarter
turn is the row every odd row's fold picks up.  :mod:`nova.biot.arcamplitude`
carries the pair from the half separation instead, and is what closes the rest.
"""

from copy import copy
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import Callable, ClassVar

import numpy as np

from nova.biot.constants import Constants
from nova.biot.incompleteelliptic import incomplete_kind
from nova.biot.matrix import Matrix


def arctan2(x1, x2):
    """Return unwraped arctan2 operator."""
    phi = np.arctan2(x1, x2)
    phi[phi < 0] += 2 * np.pi
    return phi


@dataclass
class Arc(Constants, Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d arc elements.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "arc"  # element name
    filament_centerline_limits: ClassVar[bool] = True

    attrs: dict[str, str] = field(default_factory=lambda: {"dl": "dl"})

    def __post_init__(self):
        """Load source and target geometry in local coordinate system."""
        super().__post_init__()
        self.rs = np.linalg.norm([self("source", "x1"), self("source", "y1")], axis=0)
        self.zs = self("source", "z1")
        self.r = np.linalg.norm([self("target", "x"), self("target", "y")], axis=0)
        self.z = self("target", "z")
        self._validate_source_geometry()

    def _validate_source_geometry(self):
        """Reject source geometry that has no unambiguous finite arc."""
        if np.any(~np.isfinite(self.rs) | (self.rs <= 0.0)):
            raise ValueError("arc source radius must be finite and positive")
        dl = np.asarray(self["dl"], dtype=float)
        if np.any(~np.isfinite(dl) | (dl <= 0.0)):
            raise ValueError("arc source dl must be finite and positive")
        coincident = np.all(
            np.asarray(self.source.start_point) == np.asarray(self.source.end_point),
            axis=1,
        )
        if np.any(coincident & ~self._complete_source_ring):
            raise ValueError("arc source coincident endpoints have ambiguous topology")
        if np.any(self._fit_leverage <= np.sqrt(np.finfo(float).eps)):
            raise ValueError(
                "arc source sweep is unresolved at floating-point precision"
            )

    @cached_property
    def phi(self):
        """Return global target toroidal angle."""
        return np.arctan2(self.target("y"), self.target("x"))

    @cached_property
    def _phi(self):
        """Return local target toroidal angle."""
        return np.arctan2(self("target", "y"), self("target", "x"))

    @cached_property
    def _raw_directed_sweep(self):
        """Return the source end's represented phase from its authored start."""
        start_x = self("source", "x1")
        start_y = self("source", "y1")
        delta_x = self("source", "x2") - start_x
        delta_y = self("source", "y2") - start_y
        cross = start_x * delta_y - start_y * delta_x
        dot = start_x**2 + start_y**2 + start_x * delta_x + start_y * delta_y
        return np.mod(np.arctan2(cross, dot), 2.0 * np.pi)

    @cached_property
    def _complete_source_ring(self):
        """Identify an authored complete ring from its represented topology.

        A complete path has the same stored endpoint to coordinate precision and
        a stored length consistent with its complete circumference.  Endpoint
        precision is assessed per Cartesian component, so a large translation in
        one coordinate cannot absorb a resolved gap in another.  This certificate
        does not widen the directed angular span of an open near-complete arc.
        """
        start = np.asarray(self.source.start_point, dtype=float)
        end = np.asarray(self.source.end_point, dtype=float)
        length = np.asarray(self.source("length"), dtype=float)
        eps = np.finfo(float).eps
        radius_scale = length[..., np.newaxis] / (2.0 * np.pi)
        component_ulp = np.maximum.reduce(
            np.broadcast_arrays(
                np.spacing(abs(start))[np.newaxis],
                np.spacing(abs(end))[np.newaxis],
                eps * radius_scale,
            )
        )
        same_represented_point = np.all(
            abs(end - start)[np.newaxis] <= 4.0 * component_ulp,
            axis=-1,
        )
        circumference = 2.0 * np.pi * self.rs
        length_scale = np.maximum(abs(length), abs(circumference))
        complete_length = abs(length - circumference) <= 128.0 * eps * length_scale
        return same_represented_point & complete_length

    @cached_property
    def _directed_sweep(self):
        """Return the source's directed sweep, including complete topology."""
        return np.where(
            self._complete_source_ring,
            2.0 * np.pi,
            self._raw_directed_sweep,
        )

    @cached_property
    def _directed_phase(self):
        """Return each target's directed phase from the authored start."""
        start_x = self("source", "x1")
        start_y = self("source", "y1")
        delta_x = self("target", "x") - start_x
        delta_y = self("target", "y") - start_y
        cross = start_x * delta_y - start_y * delta_x
        dot = start_x**2 + start_y**2 + start_x * delta_x + start_y * delta_y
        return np.mod(np.arctan2(cross, dot), 2.0 * np.pi)

    @cached_property
    def _fit_leverage(self):
        """Return the independent geometric leverage of the circle fit.

        A clustered minor arc resolves its centre through the midpoint
        sagitta, ``2 sin(sweep / 4)^2`` relative to the radius.  Near a complete
        turn, the endpoint chord ``sin(sweep / 2)`` is instead the limiting
        independent direction.  The smaller dimension controls the
        three-point circumcircle fit.
        """
        sweep = self._directed_sweep
        sagitta_ratio = 2.0 * np.sin(sweep / 4.0) ** 2
        endpoint_chord_ratio = abs(np.sin(sweep / 2.0))
        leverage = np.minimum(sagitta_ratio, endpoint_chord_ratio)
        return np.where(self._complete_source_ring, 1.0, leverage)

    @cached_property
    def _fit_condition(self):
        """Bound direct and fit-amplified coordinate round-off."""
        return 1.0 + 1.0 / self._fit_leverage

    @cached_property
    def _geometry_tolerance(self):
        """Return a fit-conditioned bound for local transform round-off."""
        target = self.target.stack("x", "y", "z")
        source_start = np.asarray(self.source.start_point)[np.newaxis]
        source_end = np.asarray(self.source.end_point)[np.newaxis]
        terms = np.broadcast_arrays(
            np.ones_like(self.r),
            abs(self.r),
            abs(self.rs),
            abs(self.z),
            abs(self.zs),
            np.max(abs(target), axis=-1),
            np.max(abs(source_start), axis=-1),
            np.max(abs(source_end), axis=-1),
        )
        scale = np.maximum.reduce(terms)
        return 32.0 * np.finfo(float).eps * scale * self._fit_condition

    @cached_property
    def _exact_authored_endpoint(self):
        """Identify targets that are stored source endpoints in global space."""
        target = self.target.stack("x", "y", "z")
        start = np.asarray(self.source.start_point)[np.newaxis]
        end = np.asarray(self.source.end_point)[np.newaxis]
        return np.all(target == start, axis=-1) | np.all(target == end, axis=-1)

    @cached_property
    def _same_source_ring(self):
        """Identify targets on the source circle within transform precision."""
        tolerance = self._geometry_tolerance
        return (abs(self.r - self.rs) <= tolerance) & (
            abs(self.z - self.zs) <= tolerance
        )

    @cached_property
    def _on_filament(self):
        """Identify targets on the directed authored span of the filament."""
        in_span = (self._directed_phase < self._directed_sweep) | (
            self._exact_authored_endpoint
        )
        return self._same_source_ring & in_span

    @cached_property
    def _same_ring_outside_span(self):
        """Identify finite same-ring targets in the excluded angular gap."""
        return self._same_source_ring & ~self._on_filament

    @cached_property
    def alpha(self):
        """Return system invariant angle alpha for start, end, and pi/2."""
        _phi = self._phi[np.newaxis]
        phi_s = np.stack(
            [
                np.zeros(self.shape, dtype=float),
                arctan2(self("source", "y2"), self("source", "x2")),
            ]
        )
        return np.concatenate(
            (
                (np.pi - (phi_s - _phi)) / 2,
                np.zeros((1,) + self.shape),
                np.pi / 2 * np.ones((1,) + self.shape),
            ),
            axis=0,
        )

    @property
    def sign_alpha(self):
        """Return sign(alpha)."""
        return np.where(self.alpha >= 0, 1, -1)

    @property
    def abs_alpha(self):
        """Return abs(alpha)."""
        return abs(self.alpha)

    def coefficent(func: Callable):
        """Return intergration coefficent evaluated from 0 to theta."""

        @wraps(func)
        def evaluate_coefficent(self):
            result = func(self)
            return result - result[-2]

        return evaluate_coefficent

    @cached_property
    def _index_mask(self):
        """Return index mask."""
        mask = np.ones_like(self.alpha, bool)
        mask[-2:] = False
        return mask

    def mask_index(func: Callable):
        """Return index function with intergral limit mask."""

        @wraps(func)
        def apply_mask(self):
            return func(self) & self._index_mask

        return apply_mask

    @property
    @mask_index
    def _index_A(self):
        """Return |alpha| <= pi/2 segment index."""
        return self.abs_alpha <= np.pi / 2

    @property
    @mask_index
    def _index_B(self):
        """Return |alpha| > pi/2 segment index."""
        return self.abs_alpha > np.pi / 2

    @property
    @mask_index
    def _index_C(self):
        return (self.abs_alpha > np.pi / 2) & (self.abs_alpha <= 3 * np.pi / 2)

    @property
    @mask_index
    def _index_D(self):
        """Return pi/2 < alpha <= 3pi/2 segment index."""
        return (self.alpha > 3 * np.pi / 2) & (self.alpha <= 2 * np.pi)

    @cached_property
    def _theta(self):
        """Return signed segment angle."""
        theta = self.alpha.copy()
        theta[self._index_A] = self.abs_alpha[self._index_A]
        theta[self._index_C] = np.pi - self.abs_alpha[self._index_C]
        theta[self._index_D] = 2 * np.pi - self.alpha[self._index_D]
        return theta

    @property
    def sign_theta(self):
        """Return sign(theta)."""
        return np.where(self._theta >= 0, 1, -1)

    @property
    def theta(self):
        """Return absolute segment angle."""
        return abs(self._theta)

    @cached_property
    def Phi(self):
        """Return system variant angle."""
        phi = np.pi - 2 * self.theta
        sign = np.where(phi >= 0, 1, -1)
        return np.where(sign * phi > 1e4 * self.eps, phi, sign * 1e4 * self.eps)

    @property
    # @coefficent
    def B2(self):
        """Return B2 coefficient."""
        return (
            self.rs - self.r
        ) ** 2 + 4 * self.rs * self.r * self.half_phi_sine_squared

    @property
    def half_phi_sine_squared(self):
        """Return sin(Phi / 2)^2 without subtracting a near-unit cosine."""
        return np.sin(self.Phi / 2) ** 2

    @property
    def radial_projection_gap(self):
        """Return rs - r cos(Phi) from the radial gap and a positive term."""
        return (self.rs - self.r) + 2 * self.r * self.half_phi_sine_squared

    @property
    def radial_square_gap(self):
        """Return rs^2 - r^2 with the square difference factored."""
        return (self.rs - self.r) * (self.rs + self.r)

    @property
    # @coefficent
    def D2(self):
        """Return D2 coefficient."""
        return self.gamma**2 + self.B2

    @property
    # @coefficent
    def G2(self):
        """Return G2 coefficient."""
        return self.gamma**2 + self.r**2 * np.sin(self.Phi) ** 2

    @property
    # @coefficent
    def beta_1(self):
        """Return beta 1 coefficient."""
        return self.radial_projection_gap / np.sqrt(self.G2)

    @property
    # @coefficent
    def beta_2(self):
        """Return beta 2 coefficient."""
        return self.gamma / np.sqrt(self.B2)

    @property
    # @coefficent
    def beta_3(self):
        """Return beta 3 coefficient."""
        return (
            self.gamma
            * self.radial_projection_gap
            / (self.r * np.sin(self.Phi) * np.sqrt(self.D2))
        )

    @cached_property
    # @coefficent
    def Cr(self):
        """Return Cr coefficient."""
        return (
            1 / 2 * self.gamma * self.a * self.ellipj["dn"] * np.cos(2 * self.theta)
            - 1
            / 6
            * np.arcsinh(self.beta_2)
            * np.cos(2 * self.theta)
            * (
                2 * self.r**2 * np.cos(2 * self.theta) ** 2
                - 3 * (self.rs**2 + self.r**2)
            )
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.beta_1)
            * (3 + np.cos(4 * self.theta))
            - 1 / 3 * self.r**2 * np.arctan(self.beta_3) * np.sin(2 * self.theta) ** 3
        )

    @cached_property
    # @coefficent
    def Cphi(self):
        """Return Cphi coefficient."""
        return (
            1 / 2 * self.gamma * self.a * self.ellipj["dn"] * -np.sin(2 * self.theta)
            - 1
            / 6
            * np.arcsinh(self.beta_2)
            * np.sin(2 * self.theta)
            * (2 * self.r**2 * np.sin(2 * self.theta) ** 2 + 3 * self.radial_square_gap)
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.beta_1)
            * -np.sin(4 * self.theta)
            - 1
            / 3
            * self.r**2
            * np.arctan(self.beta_3)
            * -(np.cos(2 * self.theta) ** 3)
        )

    @property
    def reps(self):
        """Return tile reps for _pi2 operator."""
        return (len(self.theta), 1, 1)

    @cached_property
    def rack2(self):
        """Return r a ck2 coefficent product."""
        return self.r * self.a * self.ck2

    @cached_property
    def theta_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the amplitude's own ``(sine, cosine)``, exact on the limit rows.

        An amplitude that lands ON a right angle is exempt from the loss, and every
        row that does is caught by the same test.  ``cos`` of a float right angle is
        6.1e-17 rather than zero, and at that amplitude the difference is not a lost
        decimal: the first kind's own step goes through it, and measured on a target
        driven onto a metre-scale ring it costs 4.1e-06 where the exact zero holds on
        round-off.  Two kinds of row reach it -- the LIMIT row, which is a right angle
        by construction and is the value every odd row's fold picks up, and an arc END
        whose target sits on that end's own plane, where the azimuthal separation
        vanishes and ``alpha = (pi - psi)/2`` lands on the float right angle exactly.

        Short of it nothing can be recovered here.  ``theta`` is folded from an
        ``alpha`` already assembled as ``(pi - psi)/2``, so the separation's own
        digits are spent before this sees the angle, and a pair formed from it is
        only as good as the angle: measured, the first kind holds 2.5e-12 for a target
        within 1e-08 radians of the end plane and within 1e-08 radii of the ring.
        :mod:`nova.biot.arcamplitude` carries the pair from the HALF separation
        instead, which is where an exact pair at an interior amplitude comes from.
        """
        amplitude = self.theta
        right_angle = amplitude >= np.pi / 2
        return (
            np.where(right_angle, 1.0, np.sin(amplitude)),
            np.where(right_angle, 0.0, np.cos(amplitude)),
        )

    @cached_property
    def ellipj(self):
        """Return the Jacobian functions at the amplitude, which are elementary.

        ``ellipj(F(theta, m), m)`` returns ``sin theta`` and ``cos theta``: the
        round trip through the first kind is the definition of the amplitude, so the
        inversion has an exact answer and is not taken.  Measured over ordinary
        geometry the two agree to 2.2e-16, and once the parameter rounds to one the
        round trip returns nothing at all -- a complete first kind of infinity has no
        Jacobian amplitude, so every one of the three comes back ``nan``.

        The third is the accuracy half.  ``dn^2 = 1 - m sin^2 theta`` is
        ``cos^2 theta + k'^2 sin^2 theta`` identically, and that is a sum of
        positives out of the exact complement where the printed form is a
        subtraction whose two terms agree to every digit on the arc's own end plane.
        It divides the toroidal field row, multiplies the radial potential row, and
        enters the composite second kind, so all three carry whatever it loses.
        """
        sine, cosine = self.theta_pair
        return {
            "sn": sine,
            "cn": cosine,
            "dn": np.sqrt(cosine**2 + self.ck2 * sine**2),
        }

    @cached_property
    def incomplete_kinds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(F, E)`` at the amplitude, off one complement-native descent.

        Both kinds come off the same arithmetic-geometric mean
        (:func:`nova.biot.incompleteelliptic.incomplete_kind`), which takes the
        modulus COMPLEMENT as its argument and the parameter alongside it.  That is
        what the first kind needs: it grows like ``-log k'``, and
        ``scipy.special.ellipkinc`` re-derives ``1 - m`` internally, so an absolute
        ``eps`` on the parameter is a relative ``eps/k'^2`` on the answer however
        exactly the parameter arrives -- supplying it alone buys a factor of six and
        no more.  Measured on the closed arc against the axisymmetric kernel, the
        flux the parameter route returns is 1.9e-05 out four micrometres off a
        metre-scale filament and infinite at forty nanometres, where this one holds on
        round-off the whole way in.

        The second kind is bounded and loses nothing either way; it comes off this
        descent because it is free once the first has run.
        """
        sine, cosine = self.theta_pair
        return incomplete_kind(
            self.theta, self.ck2, sine=sine, cosine=cosine, parameter=self.k2
        )

    @cached_property
    def Kinc(self):
        """Return end point stacked incomplete elliptic intergral of the 1st kind."""
        return self.incomplete_kinds[0]

    @cached_property
    def Einc(self):
        """Return end point stacked incomplete elliptic intergral of the 2nd kind."""
        return self.incomplete_kinds[1]

    @cached_property
    def Pi_inc(self) -> dict[int, np.ndarray]:
        """Return end point stacked incomplete elliptic intergral of the 3rd kind.

        The pole and the modulus complement are supplied from the geometry, which
        routes the evaluation through
        :func:`nova.biot.incompleteelliptic.incomplete_pole` -- the arrangement in
        which every term is positive and a near pole is reflected onto its far
        partner.  Without them the characteristic and the parameter are all the
        symmetric forms have, and ``1 - n`` at a quarter-turn amplitude has nothing
        left to dilute it: measured on a target four micrometres off a metre-scale
        ring and on the arc's end plane, the two bounded characteristics are 2.5e-04
        and 8.2e-04 out, reaching 81 % half a micrometre in -- and at forty nanometres
        the routine fails its own parameter assertion, the parameter having rounded to
        one with nothing left to form the complement from.
        """
        sine, cosine = self.theta_pair
        return {
            p: self.ellippinc(
                self.np2[p],
                self.theta,
                self.k2,
                sine=sine,
                cosine=cosine,
                pole=self.np2_pole[p],
                complement=self.ck2,
            )
            for p in range(1, 4)
        }

    @cached_property
    def Winc(self):
        """Return end point stacked composite incomplete elliptic intergral.

        ``E - k^2 sn cn/dn``, so it carries the radical's arrangement into both field
        rows that weight it.  The correction vanishes at a quarter-turn amplitude,
        where the cosine is exactly zero, which is what keeps the composite bounded
        as the target reaches the source ring and the radical goes with it.
        """
        ellipj = self.ellipj
        return self.Einc - self.k2 * ellipj["sn"] * ellipj["cn"] / ellipj["dn"]

    def _pole_gap(self, p: int) -> np.ndarray:
        """Return ``k^2 - n`` for characteristic ``p``, as a difference of poles.

        ``k^2 = 1 - k'^2`` and ``n = 1 - (1 - n)``, so the difference is
        ``(1 - n) - k'^2`` exactly -- both of them quantities the geometry supplies
        without subtracting from unity.  It sets which of the elementary branches
        below applies and divides all three of them, and it is the difference of two
        quantities that are each of order the squared distance to a source corner:
        taken from the parameter and the characteristic instead, both terms are of
        order one and the difference comes back at ``eps`` however small the geometry
        makes it.
        """
        return self.np2_pole[p] - self.ck2

    def _pole_denominator(self, p: int) -> np.ndarray:
        """Return ``1 - n sin^2 theta`` for characteristic ``p``, without subtracting.

        It is ``cos^2 theta + (1 - n) sin^2 theta`` identically, a sum of positives.
        The printed form reaches the last bits of the characteristic at a
        quarter-turn amplitude -- where the sine's square is one and nothing is left
        to dilute it -- and goes NEGATIVE there for a characteristic within an ``eps``
        of one, which puts the logarithms below at ``nan``.
        """
        sine, cosine = self.theta_pair
        return cosine**2 + self.np2_pole[p] * sine**2

    @cached_property
    def Ip(self) -> dict[int, np.ndarray]:
        """Return I(np2) coefficent.

        Four elementary branches on where the ring denominator's root sits relative
        to the range and to the modulus, selected on the sign of the pole gap
        (:meth:`_pole_gap`) so the branch and the radicals it takes can never
        disagree.  A vanishing gap -- the root ON the modulus -- is the confluence
        the two logarithmic branches close onto, and there the whole expression is
        the reciprocal radical.
        """
        zeros = np.zeros_like(self.k2)
        ones = np.ones_like(self.k2)
        Ip = {p: np.zeros_like(self.k2) for p in range(1, 4)}
        for p in Ip:
            gap = self._pole_gap(p)
            denominator = self._pole_denominator(p)
            sign = self.np2_pole[p] >= 1  # n <= 0: the root past the near END of range
            root = np.sqrt(gap, ones.copy(), where=sign)
            span = root + np.sqrt(abs(self.np2[p])) * self.ellipj["dn"]
            Ip[p] = np.where(
                sign,
                -np.sqrt(abs(self.np2[p]))
                / (2 * root)
                # the log's argument is printed as the SQUARED DIFFERENCE of the two
                # roots over the denominator, and that difference collapses to
                # nothing at a small amplitude and a far root -- the two agree to
                # every digit, and past a pole of 1e20 the printed form returns
                # exactly zero and the logarithm diverges.  Their difference times
                # their sum is
                #     (k^2 - n) - |n| dn^2 = k^2 (cos^2 theta + (1 - n) sin^2 theta)
                # identically, which is the denominator again, so the ratio is a
                # quotient of positives with the cancellation taken out
                * np.log(self.k2**2 * denominator / span**2),
                Ip[p],
            )
            Ip[p] = np.where(
                (sign := self.np2_pole[p] < 1) & (gap == 0),
                1 / self.ellipj["dn"],
                Ip[p],
            )
            Ip[p] = np.where(
                sign & (diff := gap < 0),  # n > k^2
                np.sqrt(self.np2[p], zeros.copy(), where=sign)
                / (2 * np.sqrt(-gap, ones.copy(), where=diff))
                * np.log(
                    (
                        np.sqrt(-gap, ones.copy(), where=diff)
                        + np.sqrt(self.np2[p], zeros.copy(), where=sign)
                        * self.ellipj["dn"]
                    )
                    ** 2
                    / denominator
                ),
                Ip[p],
            )
            Ip[p] = np.where(
                sign & (diff := gap > 0),  # n < k^2
                -np.sqrt(self.np2[p], zeros.copy(), where=sign)
                / (2 * np.sqrt(gap, ones.copy(), where=diff))
                * np.arcsin(
                    2
                    * np.sqrt(self.np2[p], zeros.copy(), where=sign)
                    * self.ellipj["dn"]
                    * np.sqrt(gap, zeros.copy(), where=diff)
                    / (self.k2 * abs(denominator))
                ),
                Ip[p],
            )
        return Ip

    def _exterior(self, _hat):
        """Index radial and toroidal fields.

        - pi/2 < |alpha| <= pi
        - pi < alpha <= 3pi/2
        - 3pi/2 < alpha <= 2pi

        """
        _hat_pi2 = np.tile(_hat[-1, np.newaxis], self.reps)
        _hat[self._index_C] = (
            2 * _hat_pi2[self._index_C]
            - self.sign_theta[self._index_C] * _hat[self._index_C]
        )
        _hat[self._index_D] = 4 * _hat_pi2[self._index_D] - _hat[self._index_D]
        return _hat

    @cached_property
    def _Ar_hat(self):
        """Return stacked local radial vector potential intergration coefficents."""
        return self.a / self.r * self.ellipj["dn"]

    @cached_property
    def _Aphi_hat(self):
        """Return stacked local toroidal vector potential intergration coefficents."""
        Aphi_hat = (
            self.a
            / self.r
            * self.sign_theta
            * ((1 - self.k2 / 2) * self.Kinc - self.Einc)
        )
        return self.sign_alpha * self._exterior(Aphi_hat)

    @property
    def _Ax_hat(self):
        """Return stacked local x-coordinate vector potential intergration constants."""
        return self._Ar_hat * np.cos(self._phi) - self._Aphi_hat * np.sin(self._phi)

    @property
    def _Ay_hat(self):
        """Return stacked local y-coordinate vector potential intergration constants."""
        return self._Ar_hat * np.sin(self._phi) + self._Aphi_hat * np.cos(self._phi)

    @property
    def _Az_hat(self):
        """Return stacked local z-coordinate vector potential intergration constants."""
        return np.zeros_like(self._Ar_hat)

    @cached_property
    def _Br_hat(self):
        """Return stacked local radial magnetic field intergration coefficents."""
        Br_hat = (
            self.sign_theta
            * self.gamma
            * (self.ck2 * self.Kinc - (1 - self.k2 / 2) * self.Winc)
        ) / self.rack2
        return self.sign_alpha * self._exterior(Br_hat)

    @cached_property
    def _Bphi_hat(self):
        """Return stacked local toroidal magnetic field intergration coefficents."""
        return (-self.gamma * self.ck2 / self.ellipj["dn"]) / self.rack2

    @property
    def _Bz_hat(self):
        """Return stacked local vertical magnetic field intergration coefficents.

        The second kind's weight is half of
        :attr:`nova.biot.constants.Constants.axial_weight`, which is ``2 r - b k^2``
        as the geometry gives it: the printed difference reaches zero as the target
        reaches the source ring, and it stands over the modulus complement, so on the
        closed arc it costs the vertical field 7.1e-09 where the geometric spelling
        holds on round-off.
        """
        Bz_hat = (
            self.sign_theta
            * (self.r * self.ck2 * self.Kinc - self.axial_weight / 2 * self.Winc)
        ) / self.rack2
        return self.sign_alpha * self._exterior(Bz_hat)

    @property
    def _Bx_hat(self):
        """Return stacked local x-coordinate magnetic field intergration constants."""
        return self._Br_hat * np.cos(self._phi) - self._Bphi_hat * np.sin(self._phi)

    @property
    def _By_hat(self):
        """Return stacked local y-coordinate magnetic field intergration constants."""
        return self._Br_hat * np.sin(self._phi) + self._Bphi_hat * np.cos(self._phi)

    def _intergrate(self, data):
        """Return intergral quantity."""
        return 1 / (4 * np.pi) * (data[0] - data[1])

    def _held_vector(self, attr: str, special: np.ndarray) -> np.ndarray:
        """Evaluate ordinary rows after holding exact confluences off the ring."""
        held = copy(self)
        for cls in type(self).mro():
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, cached_property):
                    held.__dict__.pop(name, None)
        held.r = np.where(special, 0.5 * self.rs, self.r)
        held.z = np.where(special, self.zs + self.rs, self.z)
        return Matrix._vector(held, attr)

    def _same_ring_limit(self, attr: str) -> np.ndarray:
        """Return the elementary excluded-gap limit in the source-local frame."""
        outside = self._same_ring_outside_span
        phase = np.where(outside, self._directed_phase, np.pi)
        sweep = np.where(outside, self._directed_sweep, 0.5 * np.pi)
        gap_from_end = phase - sweep
        gap_to_start = 2.0 * np.pi - phase
        logarithm = -np.log(np.tan(gap_from_end / 4.0)) - np.log(
            np.tan(gap_to_start / 4.0)
        )
        if attr == "A":
            radial = (np.sin(gap_to_start / 2.0) - np.sin(gap_from_end / 2.0)) / (
                2.0 * np.pi
            )
            toroidal = (
                logarithm
                - 2.0 * (np.cos(gap_to_start / 2.0) + np.cos(gap_from_end / 2.0))
            ) / (4.0 * np.pi)
            local = np.stack(
                [
                    radial * np.cos(phase) - toroidal * np.sin(phase),
                    radial * np.sin(phase) + toroidal * np.cos(phase),
                    np.zeros_like(radial),
                ],
                axis=-1,
            )
        else:
            radius = np.where(outside, self.r, 1.0)
            local = np.stack(
                [
                    np.zeros_like(logarithm),
                    np.zeros_like(logarithm),
                    logarithm / (8.0 * np.pi * radius),
                ],
                axis=-1,
            )
        return self.loc.rotate(local, "to_global")

    def _vector(self, attr: str):
        """Return finite gap limits and explicit filament singularities."""
        if not self.filament_centerline_limits:
            return Matrix._vector(self, attr)
        on_filament = self._on_filament
        outside = self._same_ring_outside_span
        ordinary = self._held_vector(attr, on_filament | outside)
        result = np.where(
            outside[..., np.newaxis], self._same_ring_limit(attr), ordinary
        )
        return np.where(on_filament[..., np.newaxis], np.nan, result)


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 12

    length = 2 * np.pi
    offset = 0

    theta = offset + np.linspace(-length / 2, length / 2, 1 + 3 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Bx", "By", "Br", "Bz", "Ay"])
    coilset.coil.insert(radius, height, 0.05, 0.05, ifttt=False, segment="arc", Ic=1e3)
    """
    for i in range(segment_number):
        coilset.winding.insert(
            points[3 * i : 1 + 3 * (i + 1)],
            {"s": (0, 0, 0.05)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=1e3,
        )
    """
    coilset.grid.solve(1500, 0.5)
    coilset.plot()

    attr = "ay"

    circle = CoilSet(field_attrs=["Bx", "By", "Br", "Bz", "Ay"])
    circle.coil.insert(
        radius, height, 0.05, 0.05, ifttt=False, segment="cylinder", Ic=1e3
    )
    circle.grid.solve(1500, 0.5)
    levels = circle.grid.plot(attr, levels=31, colors="C0", linestyles="--")

    # levels = 31
    coilset.grid.plot(attr, colors="C1", levels=levels)

    """
    # coilset.subframe.vtkplot()

    coilset.saloc["Ic"] = 5.3e5
    levels = coilset.grid.plot("ay", levels=21, nulls=False, colors="C2")
    axes = coilset.grid.axes

    segment_number = 81

    theta = np.linspace(theta[0], theta[-1], 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    line = CoilSet(field_attrs=["Bz", "Ay"])
    for i in range(segment_number):
        line.winding.insert(
            points[2 * i : 1 + 2 * (i + 1)],
            {"s": (0, 0, 0.05)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=5.3e5,
        )

    line.grid.solve(2500, 0.5)
    line.grid.plot("ay", colors="C3", linestyles="--", levels=levels)
    """
