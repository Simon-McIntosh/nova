"""Biot-Savart intergration constants."""

from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import ClassVar

import numpy as np
import scipy.special


# pylint: disable=no-member  # disable scipy.special module not found
# pylint: disable=W0631  # disable short names


def unit_nudge(limit_factor=1.5, threshold_factor=3):
    """
    Nudge output to avoid unit singularities.

    Parameters
    ----------
    limit_factor : float, optional
        Limit factor multiplies the class eps such that as output tends to
        unit the result tends to limit * self.eps. The default is 1.5.

    threshold_factor : float, optional
        Threshold factor above which linear nudging is applied.
        This factor multiplies the class eps such that the transform is
        applied when output > 1 - thershold * self.eps.
        The default is None .

    Raises
    ------
    ValueError
        When limit_factor > threshold_factor.

    Returns
    -------
    Nudged output.

    """
    if (
        threshold_factor is not None
        and limit_factor is not None
        and limit_factor > threshold_factor
    ):
        raise ValueError(
            f"limit_factor > threshold_factor {limit_factor} > {threshold_factor}"
        )

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            output = method(self, *args, **kwargs)

            def defactor(factor, eps, default):
                if factor is None:
                    return default
                return factor * eps

            limit = defactor(limit_factor, self.eps, 0)
            threshold = defactor(threshold_factor, self.eps, 1)
            delta = output - (1 - threshold)
            unit_delta = delta / threshold
            return np.where(
                (output < 1 + limit) & (delta > 0),
                (1 - threshold) + unit_delta * (threshold - limit),
                output,
            )

        return wrapper

    return decorator


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
    def k2(self):
        """Return k2 coefficient."""
        return (1 - self.eps) * 4 * self.r * self.rs / self.a2

    @property
    def ck2(self):
        """Return complementary modulus."""
        return 1 - self.k2

    @cached_property
    def v(self):
        """Return v coefficient."""
        return 1 + self.k2 * (self.gamma**2 - self.b * self.r) / (2 * self.r * self.rs)

    @cached_property
    def K(self):
        """Return complete elliptic intergral of the 1st kind."""
        return self.ellipk(self.k2)

    @cached_property
    def E(self):
        """Return complete elliptic intergral of the 2nd kind."""
        return self.ellipe(self.k2)

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
        rf = scipy.special.elliprf(x, y, z)
        # rf[(y == 0) | (y == 1)] = scipy.special.elliprc(0, 1)  #
        return rf

    @classmethod
    def elliprj(cls, x, y, z, p):
        """Retrun symmetric elliptic integral of the third kind."""
        rj = scipy.special.elliprj(x, y, z, p)
        # rj[y == p] = scipy.special.elliprd(x[y == p], z[y == p], p[y == p])  #
        # rj[p == 1] = scipy.special.elliprd(x[p == 1], y[p == 1], 1)  #
        return rj

    @classmethod
    def ellipp(cls, n, m):
        """
        Return complete elliptic intergral of the 3rd kind.

        Adapted from https://github.com/scipy/scipy/issues/4452.
        """
        x, y, z, p = np.zeros_like(n), 1 - m, np.ones_like(n), 1 - n
        rf = cls.elliprf(x, y, z)
        rj = cls.elliprj(x, y, z, p)
        return rf + rj * n / 3

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
    def ellippinc(cls, n, phi, m, *, sine=None, cosine=None):
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
        as ``phi - t pi`` instead loses the residual to cancellation, and the
        floor-and-offset count it used to be taken with — ``(phi + pi/2)//pi`` —
        rounds up one step short of a quarter turn, putting the residual an ulp
        OUTSIDE the closed quarter it was then checked against: an amplitude one
        representable step below a right angle, which is where a target on an
        arc's own end plane lands.

        ``sine`` and ``cosine`` supply the amplitude's pair where the caller
        formed it from its geometry, as
        :func:`nova.biot.incompleteelliptic.incomplete_pole` and
        :mod:`nova.biot.arcamplitude` do; near a quarter turn a pair carried
        exactly is the difference between a relative accuracy and an absolute
        one.  Adapted from https://github.com/scipy/scipy/issues/4452.
        """
        phi = np.asarray(phi, dtype=float)
        sine = np.sin(phi) if sine is None else np.asarray(sine)
        cosine = np.cos(phi) if cosine is None else np.asarray(cosine)
        turns = np.round(phi / np.pi)
        parity = 1.0 - 2.0 * abs(np.remainder(turns, 2.0))
        sine, cosine = parity * sine, parity * cosine
        assert np.all(m < 1)
        # the fold leaves the residual amplitude inside the closed quarter, so
        # its cosine is non-negative -- to within the resolution with which a
        # float amplitude locates a half turn at all, which is what the old
        # bound on the subtracted angle could not honour
        assert np.all(cosine >= -4 * cls.eps * np.maximum(1.0, abs(phi)))
        folded = cls._ellippinc(n, sine, cosine, m)
        if not np.any(turns):  # a quarter-turn range never leaves the branch
            return folded
        return 2 * turns * cls.ellipp(n, m) + folded

    # @unit_nudge()
    def _np2_2(self):
        return 2 * self.r / (self.r + self.c)

    # @unit_nudge()
    def _np2_3(self):
        return 4 * self.r * self.rs / self.b**2

    @cached_property
    def np2(self) -> dict[int, np.ndarray]:
        """Return np**2 constant."""
        return {
            1: 2 * self.r / (self.r - self.c - self.eps),
            2: (1 - self.eps) * 2 * self.r / (self.r + self.c),
            3: (1 - self.eps) * 4 * self.r * self.rs / self.b**2,
        }

    @property
    def Qr(self) -> dict[int, np.ndarray]:
        """Return Qr(p) coefficient."""
        Qr = {
            p: (self.rs - (-1) ** p * self.c)
            * self.np2[p]
            * self.gamma**2
            * self.c
            / self.r
            for p in [1, 2]
        }
        Qr[3] = np.zeros_like(self.r)
        return Qr

    @property
    def Qphi(self) -> dict[int, np.ndarray]:
        """Return Qphi(p) coefficient."""
        Qphi = {
            p: (self.rs - (-1) ** p * self.c) * (-1) ** p * (self.c**2 + self.gamma**2)
            for p in [1, 2]
        }
        Qphi[3] = np.zeros_like(self.r)
        return Qphi

    @property
    def Qz(self) -> dict[int, np.ndarray]:
        """Return Qz(p) coefficient."""
        Qz = {
            p: (self.rs - (-1) ** p * self.c) * -2 * self.gamma * self.c * self.np2[p]
            for p in [1, 2]
        }
        Qz[3] = self.gamma * self.b * (self.rs - self.r) * self.np2[3]
        return Qz

    @cached_property
    def Pr(self) -> dict[int, np.ndarray]:
        """Return Pr(p) coefficient."""
        Pr = {
            p: (self.rs - (-1) ** p * self.c) * (-1) ** p * (self.c**2 + 5 * self.r**2)
            for p in [1, 2]
        }
        Pr[3] = -self.rs * (self.rs**2 + 3 * self.r**2)
        return Pr

    @cached_property
    def Pphi(self) -> dict[int, np.ndarray]:
        """Return Pphi(p) coefficient."""
        Pphi = {
            p: (self.rs - (-1) ** p * self.c)
            * self.np2[p]
            * self.c
            * (3 * self.r**2 - self.c2)
            / (2 * self.r)
            for p in [1, 2]
        }
        Pphi[3] = -self.rs / self.b * (self.rs - self.r) * (3 * self.r**2 - self.rs**2)
        return Pphi

    @cached_property
    def Pi(self) -> dict[int, np.ndarray]:
        """Return complete elliptc intergral of the 3rd kind."""
        return {p: self.ellipp(self.np2[p], self.k2) for p in range(1, 4)}

    def p_sum(self, func_a, func_b):
        """Return p sum."""
        result = np.zeros_like(func_b[1])
        for p in range(1, 4):
            result += (-1) ** p * func_a[p] * func_b[p]
        return result
