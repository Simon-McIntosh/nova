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

The three kinds are therefore routed through the complement-native descents,
:mod:`nova.biot.completeelliptic` over the whole quarter range and
:mod:`nova.biot.incompleteelliptic` at an interior amplitude, rather than through
a parameter form that has to re-derive what the caller already knows exactly.
The third kind's entry points here take the pole and the complement alongside the
characteristic and the modulus, so a caller that formed them from its geometry
loses nothing on the way in.

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

from nova.biot.completeelliptic import complete_kind, complete_pole
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
        """Return v coefficient."""
        return 1 + self.k2 * (self.gamma**2 - self.b * self.r) / (2 * self.r * self.rs)

    @cached_property
    def _complete_kinds(self):
        """Return ``(K, E)`` off ONE descent, from the modulus complement.

        The two share the descent and differ only in the weight on ``sin^2``, so
        the second kind costs a handful of multiplies rather than a second
        iteration -- which is worth having because both rows use both.
        """
        return complete_kind(self.ck2)

    @cached_property
    def K(self):
        """Return complete elliptic intergral of the 1st kind."""
        return self._complete_kinds[0]

    @cached_property
    def E(self):
        """Return complete elliptic intergral of the 2nd kind."""
        return self._complete_kinds[1]

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

    @staticmethod
    def _complements(n, m, pole, complement):
        """Return the pole and the modulus complement, preferring what was supplied.

        ``pole`` is the third kind's denominator at the far end of the range,
        ``1 - n``, and ``complement`` is ``k'^2``.  A caller holding the geometry
        forms both exactly; forming them here instead caps the integral at ``eps``
        over whichever is small, which at a corner is both.
        """
        pole = 1.0 - np.asarray(n) if pole is None else np.asarray(pole)
        complement = (
            1.0 - np.asarray(m) if complement is None else np.asarray(complement)
        )
        return pole, complement

    @classmethod
    def ellipp(cls, n, m, *, pole=None, complement=None):
        """Return complete elliptic intergral of the 3rd kind."""
        pole, complement = cls._complements(n, m, pole, complement)
        return complete_pole(pole, complement)

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
        formed it from its geometry, as
        :func:`nova.biot.incompleteelliptic.incomplete_pole` and
        :mod:`nova.biot.arcamplitude` do; near a quarter turn a pair carried
        exactly is the difference between a relative accuracy and an absolute
        one.  ``pole`` and ``complement`` do the same for the characteristic and
        the modulus -- see :meth:`_complements`.
        """
        phi = np.asarray(phi, dtype=float)
        sine = np.sin(phi) if sine is None else np.asarray(sine)
        cosine = np.cos(phi) if cosine is None else np.asarray(cosine)
        pole, complement = cls._complements(n, m, pole, complement)
        turns = np.round(phi / np.pi)
        parity = 1.0 - 2.0 * abs(np.remainder(turns, 2.0))
        sine, cosine = parity * sine, parity * cosine
        # a vanishing complement is a target ON the source ring, which the descent
        # carries as its own confluence; a negative one is not a modulus at all
        assert np.all(complement >= 0)
        # the fold leaves the residual amplitude inside the closed quarter, so
        # its cosine is non-negative -- to within the resolution with which a
        # float amplitude locates a half turn at all
        assert np.all(cosine >= -4 * cls.eps * np.maximum(1.0, abs(phi)))
        folded = incomplete_pole(pole, complement, sine, cosine)
        if not np.any(turns):  # a quarter-turn range never leaves the branch
            return folded
        return 2 * turns * complete_pole(pole, complement) + folded

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
