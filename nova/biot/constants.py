"""Ring integration constants for the Biot-Savart arc reduction.

:class:`Constants` holds the quantities a circular source ring and a target
share -- the modulus, the three ring characteristics, and the coefficient
families that weight them -- for the elements built on it: the complete circular
filament (:class:`nova.biot.circle.Circle`), the finite arc
(:class:`nova.biot.arc.Arc`) and the arc thickened to a rectangular section
(:class:`nova.biot.bow.Bow`).

**Every pole and the modulus are taken as their exact geometric complement.**
That is the file's organising rule and it is not tidiness.  The complement of the
modulus is the target's squared distance to the source ring over the squared ring
span, and the complement of a ring characteristic -- the denominator's value at
the far end of the range -- is likewise an exact square of the geometry.  Both
vanish at a corner, and the elliptic integrals they feed grow there: the first
kind like ``-log k'``, the third like the inverse root of its pole.  Formed as
``1 - k^2`` or ``1 - n`` each is known only to an absolute ``eps``, so the
integral is capped at ``eps`` over the complement -- a centimetre off a
metre-scale ring puts that at 1e-06, and a target on a section face at nothing at
all.  The identities that make the geometric spelling available are

    r - c = -gamma^2/(r + c)                       for c = sqrt(gamma^2 + r^2)
    1 - n1 = ((r + c)/gamma)^2      1 - n2 = (gamma/(r + c))^2 = 1/(1 - n1)
    1 - n3 = ((rs - r)/b)^2         k'^2   = (gamma^2 + (rs - r)^2)/a^2

and each right-hand side is a sum or a square of positives.  The near pole's
numerator ``rs - c`` goes the same way: it is ``(rs - r)`` less
``gamma^2/(r + c)``, both exact, where the printed difference keeps no digits on
a corner's own radius.

So every argument here is a complement, and each kind is taken by whichever route
accepts one.  The first kind has a Cephes entry point on the complement,
``ellipkm1``, and the second is bounded and takes the parameter without loss, so
those two stay where they are.  The THIRD is the one that needs a routine of its
own -- there is no complement-native library form, and the printed
``R_F + (n/3) R_J`` puts two nearly-equal terms of opposite sign against each
other once the pole is past the range -- so its entry points here take the pole
and the complement alongside the characteristic and the modulus, and a caller
that supplies them is routed through :mod:`nova.biot.completeelliptic` over the
whole quarter range and :mod:`nova.biot.incompleteelliptic` at an interior
amplitude.  :attr:`Constants.Pi` supplies them from :attr:`Constants.np2_pole`;
the arc rows are the callers that still have to.

A caller that cannot is routed through Carlson's forms as before, and that is a
DOMAIN split rather than a fallback: the descent's pole must be above zero -- the
ring denominator's root outside the range -- which every characteristic a source
ring produces satisfies, while the general entry points also carry a
characteristic above one, where the root falls inside the range and the integral
is a principal value the descent has no branch for.

Measured on a compute node over two million elements: the descent costs the third
kind 544 ns an element against Carlson's 664, and would cost the first and second
pair 1245 against Cephes' 290, which is why the split runs where it does.

One confluence is not reached by the complements alone.  A target LEVEL with a
source corner puts both roots of the ring denominator on the ends of the range:
the first characteristic diverges as ``gamma^-2`` while its integral vanishes as
``|gamma|``, so their product is bounded and ODD, with one-sided limits of
``-/+ pi r``.  ``gamma`` is held at one inside that characteristic and its pole so
every array stays finite, and the explicit powers of ``gamma`` the coefficient
families carry put the product at zero on the level -- the mean of the two limits,
which is the assignment that leaves the reduction continuous across its own
corner planes.  Off the level nothing is held and the complements are exact.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
import scipy.special

from nova.biot.completeelliptic import complete_pole
from nova.biot.incompleteelliptic import incomplete_pole


# pylint: disable=no-member  # disable scipy.special module not found
# pylint: disable=W0631  # disable short names


@dataclass
class Constants:
    """Manage biot intergration constants."""

    rs: np.ndarray = field(default_factory=lambda: np.zeros_like([]))
    zs: np.ndarray = field(default_factory=lambda: np.zeros_like([]))
    r: np.ndarray = field(default_factory=lambda: np.zeros_like([]))
    z: np.ndarray = field(default_factory=lambda: np.zeros_like([]))

    eps: ClassVar[np.float64] = 2 * np.finfo(float).eps

    def sign(self, x):
        """Return sign of array -1 if x < 0 else 1."""
        return np.where(abs(x) > 1e4 * self.eps, np.sign(x), 0)

    @cached_property
    def gamma(self):
        """Return gamma coefficient."""
        return self.zs - self.z

    @property
    def a2(self):
        """Return a**2 coefficient."""
        return self.gamma**2 + (self.rs + self.r) ** 2

    @property
    def a(self):
        """Return a coefficient."""
        return np.sqrt(self.a2)

    @property
    def b(self):
        """Return b coefficient."""
        return self.rs + self.r

    @property
    def c2(self):
        """Return c**2 coefficient."""
        return self.gamma**2 + self.r**2

    @property
    def c(self):
        """Return c coefficient."""
        return np.sqrt(self.c2)

    @cached_property
    def radius_sum(self):
        """Return r + c, which every ring pole and the near numerator are built on.

        The ring denominator's two roots sit either side of the range at distances
        set by ``r - c``, and that difference is ``-gamma^2/(r + c)`` exactly.  So
        the sum is the quantity the geometry supplies and the difference is never
        formed.
        """
        return self.r + self.c

    @cached_property
    def k2(self):
        """Return the squared modulus, ``4 r rs/a^2``."""
        return 4 * self.r * self.rs / self.a2

    @cached_property
    def ck2(self):
        """Return the modulus complement, from the geometry rather than from ``k2``.

        The target's squared distance to the source ring over the squared ring
        span: the two radicals ``a^2`` and ``a^2 - 4 r rs`` differ by exactly
        ``4 r rs``, so the complement is a sum of squares over one of them and
        needs no subtraction from unity.  It divides the field rows, and the first
        kind's accuracy is bounded by it, so an absolute ``eps`` here is a relative
        ``eps/k'^2`` on the answer.
        """
        return (self.gamma**2 + (self.rs - self.r) ** 2) / self.a2

    @cached_property
    def v(self):
        """Return v coefficient, which VANISHES at a source corner.

        ``1 + k^2 (gamma^2 - b r)/(2 r rs)`` collapses onto

            v = (3 gamma^2 + (rs - r)(rs + r))/a^2

        exactly, and that is the arrangement to take it in: the printed form is one
        plus a term that reaches minus one, so it comes back at order the squared
        distance to the corner from two quantities of order one, where every term
        of the collapsed numerator is a product of exact differences.  It weights
        the first kind against the second in the radial row, so the loss is damped
        by the small factor it carries -- but it is 3.2e-10 at a corner six
        micrometres from a metre-scale target and free to avoid.
        """
        return (3 * self.gamma**2 + (self.rs - self.r) * self.b) / self.a2

    @cached_property
    def K(self):
        """Return complete elliptic intergral of the 1st kind, from the complement.

        ``ellipkm1`` takes ``k'^2`` directly, so the sensitive kind -- it grows like
        ``-log k'`` and is therefore only as good as its argument -- never sees the
        parameter at all.  The complement-native descent
        (:func:`nova.biot.completeelliptic.complete_kind`) agrees with it to a
        couple of ulp and costs four times as much for this pair, so it is what the
        THIRD kind is taken through and not this one.

        A vanishing complement is a target ON the source ring, where this is the
        divergence rather than a number -- the same convention the axisymmetric
        filament kernel returns, and the right one for a filament, which has been
        asked a question it cannot answer.  The finite-section reduction takes its
        first kind through the descent's finite part instead.
        """
        return scipy.special.ellipkm1(self.ck2)

    @cached_property
    def E(self):
        """Return complete elliptic intergral of the 2nd kind.

        Held at the parameter's own limit, because ``4 r rs`` and ``a^2`` are formed
        independently and their ratio can land an ulp ABOVE one for a target within
        about ``1e-08`` ring radii of the source.  ``E`` is bounded and smooth
        through ``m = 1``, where it is exactly one, so holding it there costs
        nothing anywhere -- and this is the only place the parameter is the argument
        of anything.
        """
        return scipy.special.ellipe(np.minimum(self.k2, 1.0))

    @cached_property
    def U(self):
        """Return U coefficient."""
        return (
            self.k2
            * (4 * self.gamma**2 + 3 * self.rs**2 - 5 * self.r**2)
            / (4 * self.r)
        )

    @staticmethod
    def _ellip(kind: str, /, *args, out=None, shape=None, where=True):
        """Return evaluation of scipy.special ellip{kind}."""
        if out is None:
            out = np.zeros_like(args[0], dtype=float, shape=shape)
        func = getattr(scipy.special, f"ellip{kind}")
        return func(*args, out=out, where=where)

    @classmethod
    def ellipkinc(cls, phi, m):
        """Return incomplete elliptic intergral of the 1st kind."""
        return cls._ellip("kinc", phi, m)

    @classmethod
    def ellipeinc(cls, phi, m):
        """Return incomplete elliptic intergral of the 2nd kind."""
        return cls._ellip("einc", phi, m)

    @classmethod
    def ellipk(cls, m):
        """Return complete elliptic intergral of the 1st kind."""
        return cls._ellip("k", m)

    @classmethod
    def ellipe(cls, m):
        """Return complete elliptic intergral of the 2nd kind."""
        return cls._ellip("e", m)

    @classmethod
    def elliprf(cls, x, y, z):
        """Return completely-symmetric elliptic integral of the first kind."""
        return scipy.special.elliprf(x, y, z)

    @classmethod
    def elliprj(cls, x, y, z, p):
        """Retrun symmetric elliptic integral of the third kind."""
        return scipy.special.elliprj(x, y, z, p)

    @staticmethod
    def _exact_arguments(pole, complement):
        """Return the pole and the modulus complement a caller supplied, or ``None``.

        ``pole`` is the third kind's denominator at the far end of the range,
        ``1 - n``, and ``complement`` is ``k'^2``.  Both or neither: the descent's
        accuracy comes from having every argument exact, and one exact argument
        beside one reached by subtraction is the subtraction's accuracy.
        """
        if pole is None or complement is None:
            return None
        return np.asarray(pole), np.asarray(complement)

    @classmethod
    def ellipp(cls, n, m, *, pole=None, complement=None):
        """Return complete elliptic intergral of the 3rd kind.

        Two routes, and which one runs is decided by whether the caller can supply
        the POLE and the modulus complement from its own geometry.

        With them, Bulirsch's descent
        (:func:`nova.biot.completeelliptic.complete_pole`): the arguments enter as
        themselves, the iteration is a sum of positives, and one expression spans
        eighteen decades of pole.  Its domain is a pole ABOVE zero -- the ring
        denominator's root outside the range -- which is every characteristic a
        source ring produces, all three of them.

        Without them, Carlson's symmetric forms on ``1 - n`` and ``1 - m``.  That is
        the general entry point rather than a fallback, because it carries a
        characteristic above one, where the root falls INSIDE the range and the
        integral is a principal value the descent has no branch for.  What it costs
        where both routes apply is the subtraction: ``1 - n`` and ``1 - m`` each
        carry an absolute ``eps``, and the two terms are of opposite sign and nearly
        equal once the pole is far past the range.
        """
        exact = cls._exact_arguments(pole, complement)
        if exact is not None:
            return complete_pole(*exact)
        n = np.asarray(n)
        x, y, z, p = np.zeros_like(n), 1 - np.asarray(m), np.ones_like(n), 1 - n
        return cls.elliprf(x, y, z) + cls.elliprj(x, y, z, p) * n / 3

    @classmethod
    def _ellippinc(cls, n, sine, cosine, m):
        """Return the 3rd kind over a quarter turn, from the amplitude's own pair.

        Carlson's form for the amplitude ``phi`` is

            Pi(n, phi, m) = sin phi R_F(cos^2 phi, 1 - m sin^2 phi, 1)
                          + (n/3) sin^3 phi R_J(..., 1 - n sin^2 phi)

        and every one of those three arguments is taken here as a SUM OF
        POSITIVES rather than as a subtraction from one.  ``cos^2 phi`` written
        as ``1 - sin^2 phi`` is the whole difficulty: within about 1e-8 of a
        quarter turn the sine rounds to one, the difference collapses to exactly
        zero, and the evaluation silently returns the COMPLETE integral in place
        of the one it was asked for.  The other two follow the same identity,
        ``1 - m sin^2 = cos^2 + (1 - m) sin^2``, which additionally keeps the
        radical's relative accuracy where the target approaches the source ring
        and both the modulus complement and the cosine are small at once.

        What it cannot do is take the characteristic's own complement from the
        caller: ``1 - n`` is formed here, and at an amplitude near a quarter turn
        the sine's square is one and nothing is left to dilute the loss.  A caller
        holding the pole exactly is served by
        :func:`nova.biot.incompleteelliptic.incomplete_pole` instead -- see
        :meth:`ellippinc`.

        Odd in the amplitude through ``sine``, so the fold's parity needs no
        separate sign.
        """
        squared_sine, squared_cosine = sine * sine, cosine * cosine
        radical = squared_cosine + (1.0 - m) * squared_sine
        unit = np.ones_like(radical)
        rf = cls.elliprf(squared_cosine, radical, unit)
        rj = cls.elliprj(
            squared_cosine, radical, unit, squared_cosine + (1.0 - n) * squared_sine
        )
        return sine * rf + n * sine * squared_sine * rj / 3.0

    @classmethod
    def ellippinc(
        cls, n, phi, m, *, sine=None, cosine=None, pole=None, complement=None
    ):
        """
        Return incomplete elliptic intergral of the 3rd kind.

        The integrand is half-turn periodic and even about zero, so its integral
        is ODD and gains two complete integrals per half turn of amplitude,

            Pi(n, t pi + d, m) = 2 t Pi(n, m) + Pi(n, d, m)

        with ``t = round(phi/pi)`` and ``|d| <= pi/2`` by construction of the
        rounding.  The residual amplitude ``d`` is never FORMED: what the
        evaluation asks for is its sine and cosine, and

            sin d = (-1)^t sin phi,   cos d = (-1)^t cos phi

        are exact, so the fold is a sign rather than a subtraction.  Taking it
        as ``phi - t pi`` instead loses the residual to cancellation, and a
        floor-and-offset count -- ``(phi + pi/2)//pi`` -- rounds up one step short
        of a quarter turn, putting the residual an ulp OUTSIDE the closed quarter
        it is then checked against: an amplitude one representable step below a
        right angle, which is where a target on an arc's own end plane lands.

        ``sine`` and ``cosine`` supply the amplitude's pair where the caller
        formed it from its geometry, as :mod:`nova.biot.arcamplitude` does; near a
        quarter turn a pair carried exactly is the difference between a relative
        accuracy and an absolute one.  ``pole`` and ``complement`` do the same for
        the characteristic and the modulus, and supplying them routes the residual
        through :func:`nova.biot.incompleteelliptic.incomplete_pole` -- which
        additionally reflects a near pole onto its far partner, the arrangement in
        which the whole expression is a sum of positives.  Without them the
        residual goes through Carlson's forms on ``1 - n``; see :meth:`ellipp` for
        why that is the general route and what the two differ in.
        """
        phi = np.asarray(phi, dtype=float)
        sine = np.sin(phi) if sine is None else np.asarray(sine)
        cosine = np.cos(phi) if cosine is None else np.asarray(cosine)
        exact = cls._exact_arguments(pole, complement)
        turns = np.round(phi / np.pi)
        parity = 1.0 - 2.0 * abs(np.remainder(turns, 2.0))
        sine, cosine = parity * sine, parity * cosine
        if exact is None:
            # the parameter route has to form the modulus complement itself, and a
            # parameter at one leaves it nothing to form it from
            assert np.all(np.asarray(m) < 1)
        # the fold leaves the residual amplitude inside the closed quarter, so
        # its cosine is non-negative -- to within the resolution with which a
        # float amplitude locates a half turn at all
        assert np.all(cosine >= -4 * cls.eps * np.maximum(1.0, abs(phi)))
        if exact is None:
            folded = cls._ellippinc(n, sine, cosine, m)
        else:
            folded = incomplete_pole(*exact, sine, cosine)
        if not np.any(turns):  # a quarter-turn range never leaves the branch
            return folded
        complete = cls.ellipp(n, m, pole=pole, complement=complement)
        return 2 * turns * complete + folded

    @cached_property
    def _held_gamma(self):
        """Return gamma, held at one where the target is LEVEL with the source.

        The first ring pole is ``((r + c)/gamma)^2`` and diverges on the level,
        where its integral vanishes at the same rate.  Holding the argument keeps
        every array finite and the derivative with it; the explicit powers of
        ``gamma`` the coefficient families carry are what put the bounded product
        at zero there, which is the mean of its two one-sided limits.
        """
        return np.where(self.gamma == 0.0, 1.0, self.gamma)

    @cached_property
    def np2(self) -> dict[int, np.ndarray]:
        """Return the three ring characteristics ``n``.

        The first is ``2 r/(r - c)``, taken through ``r - c = -gamma^2/(r + c)``:
        the printed difference has both terms agreeing to every digit on a source
        corner's own plane, where this spelling is exact and diverges as it should.
        """
        return {
            1: -2 * self.r * self.radius_sum / self._held_gamma**2,
            2: 2 * self.r / self.radius_sum,
            3: 4 * self.r * self.rs / self.b**2,
        }

    @cached_property
    def np2_pole(self) -> dict[int, np.ndarray]:
        """Return each characteristic's complement, the third kind's own argument.

        The denominator's value at the far end of the range, ``1 - n``, as the
        exact square the geometry gives it.  The first two are reciprocal --
        ``(1 - n1)(1 - n2) = 1``, the ring denominator's two roots sitting
        symmetrically about the range -- and a pair formed by subtraction loses
        that identity along with the poles.
        """
        return {
            1: (self.radius_sum / self._held_gamma) ** 2,
            2: (self.gamma / self.radius_sum) ** 2,
            3: ((self.rs - self.r) / self.b) ** 2,
        }

    @cached_property
    def edge(self) -> dict[int, np.ndarray]:
        """Return ``rs -/+ c``, the numerator each ring root carries.

        The far root's numerator is a sum and needs nothing.  The near root's is
        ``rs - c``, which cancels to nothing when the target sits on this ring's
        own radius: it is ``(rs - r)`` less ``gamma^2/(r + c)``, both exact, where
        the printed difference is down to the last bits of the radius' own spacing
        a rounding error off the corner.
        """
        return {
            1: self.rs + self.c,
            2: (self.rs - self.r) - self.gamma**2 / self.radius_sum,
        }

    @property
    def Qr(self) -> dict[int, np.ndarray]:
        """Return Qr(p) coefficient."""
        Qr = {
            p: self.edge[p] * self.np2[p] * self.gamma**2 * self.c / self.r
            for p in [1, 2]
        }
        Qr[3] = np.zeros_like(self.r)
        return Qr

    @property
    def Qphi(self) -> dict[int, np.ndarray]:
        """Return Qphi(p) coefficient."""
        Qphi = {p: self.edge[p] * (-1) ** p * (self.c2 + self.gamma**2) for p in [1, 2]}
        Qphi[3] = np.zeros_like(self.r)
        return Qphi

    @property
    def Qz(self) -> dict[int, np.ndarray]:
        """Return Qz(p) coefficient."""
        Qz = {p: self.edge[p] * -2 * self.gamma * self.c * self.np2[p] for p in [1, 2]}
        Qz[3] = self.gamma * self.b * (self.rs - self.r) * self.np2[3]
        return Qz

    @cached_property
    def Pr(self) -> dict[int, np.ndarray]:
        """Return Pr(p) coefficient."""
        Pr = {p: self.edge[p] * (-1) ** p * (self.c2 + 5 * self.r**2) for p in [1, 2]}
        Pr[3] = -self.rs * (self.rs**2 + 3 * self.r**2)
        return Pr

    @cached_property
    def Pphi(self) -> dict[int, np.ndarray]:
        """Return Pphi(p) coefficient.

        ``3 r^2 - c^2`` is ``2 r^2 - gamma^2``, the same two terms without the
        target radius appearing in both of them.
        """
        moment = self.c * (2 * self.r**2 - self.gamma**2) / (2 * self.r)
        Pphi = {p: self.edge[p] * self.np2[p] * moment for p in [1, 2]}
        Pphi[3] = -self.rs / self.b * (self.rs - self.r) * (3 * self.r**2 - self.rs**2)
        return Pphi

    @cached_property
    def Pi(self) -> dict[int, np.ndarray]:
        """Return complete elliptc intergral of the 3rd kind."""
        return {p: complete_pole(self.np2_pole[p], self.ck2) for p in range(1, 4)}

    def p_sum(self, func_a, func_b):
        """Return p sum."""
        result = np.zeros_like(func_b[1])
        for p in range(1, 4):
            result += (-1) ** p * func_a[p] * func_b[p]
        return result
