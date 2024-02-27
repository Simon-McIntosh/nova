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
            "limit_factor > threshold_factor " f"{limit_factor} > {threshold_factor}"
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
    def phi_(self):
        """Return system variant angle."""
        phi = np.pi - 2 * self.alpha
        return np.where(abs(phi) > 1e4 * self.eps, phi, 1e4 * self.eps)

    @property
    def B2(self):
        """Return B2 coefficient."""
        return self.rs**2 + self.r**2 - 2 * self.r * self.rs * np.cos(self.phi_)

    @property
    def D2(self):
        """Return D2 coefficient."""
        return self.gamma**2 + self.B2

    @property
    def G2(self):
        """Return G2 coefficient."""
        return self.gamma**2 + self.r**2 * np.sin(self.phi_) ** 2

    @property
    def beta_1(self):
        """Return beta 1 coefficient."""
        return (self.rs - self.r * np.cos(self.phi_)) / np.sqrt(self.G2)

    @property
    def beta_2(self):
        """Return beta 2 coefficient."""
        return self.gamma / np.sqrt(self.B2)

    @property
    def beta_3(self):
        """Return beta 3 coefficient."""
        return (
            self.gamma
            * (self.rs - self.r * np.cos(self.phi_))
            / (self.r * np.sin(self.phi_) * np.sqrt(self.D2))
        )

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
    def ellippi(cls, n, m):
        """
        Return complete elliptic intergral of the 3rd kind.

        Adapted from https://github.com/scipy/scipy/issues/4452.
        """
        """
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = cls._ellip('rf', x, y, z, shape=m.shape) #, where=(m < 1))
        #cls._ellip('rc', 0, 1, out=rf, where=np.isclose(y, 1))
        rj = cls._ellip('rj', x, y, z, p, shape=m.shape) #, where=(m < 1))
        #cls._ellip('rd', x, z, p, out=rj, where=np.isclose(y, p))
        #cls._ellip('rd', x, y, p, out=rj, where=np.isclose(p, 1))
        return rf + rj * n / 3
        """
        x, y, z, p = 0, 1 - m, 1, 1 - n
        rf = scipy.special.elliprf(x, y, z)
        rf[(y == 0) | (y == 1)] = scipy.special.elliprc(0, 1)
        rj = scipy.special.elliprj(x, y, z, p)
        rj[y == p] = scipy.special.elliprd(x, z, p[y == p])
        rj[p == 1] = scipy.special.elliprd(x, y[p == 1], 1)
        return rf + rj * n / 3

    @cached_property
    def Cr(self):
        """Return Cr coefficient."""
        return (
            1
            / 2
            * self.gamma
            * self.a
            * np.sqrt(1 - self.k2 * np.sin(self.alpha) ** 2)
            * np.cos(2 * self.alpha)
            - 1
            / 6
            * np.arcsinh(self.beta_2)
            * np.cos(2 * self.alpha)
            * (
                2 * self.r**2 * np.cos(2 * self.alpha) ** 2
                - 3 * (self.rs**2 + self.r**2)
            )
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.beta_1)
            * (3 + np.cos(4 * self.alpha))
            - 1 / 3 * self.r**2 * np.arctan(self.beta_3) * np.sin(2 * self.alpha) ** 3
        )

    @cached_property
    def Cphi(self):
        """Return Cphi coefficient."""
        return (
            1
            / 2
            * self.gamma
            * self.a
            * np.sqrt(1 - self.k2 * np.sin(self.alpha) ** 2)
            * -np.sin(2 * self.alpha)
            - 1
            / 6
            * np.arcsinh(self.beta_2)
            * np.sin(2 * self.alpha)
            * (
                2 * self.r**2 * np.sin(2 * self.alpha) ** 2
                + 3 * (self.rs**2 - self.r**2)
            )
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.beta_1)
            * -np.sin(4 * self.alpha)
            - 1 / 3 * self.r**2 * np.arctan(self.beta_3) * -np.cos(2 * self.alpha) ** 3
        )

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
        return {p: self.ellippi(self.np2[p], self.k2) for p in range(1, 4)}

    def p_sum(self, func_a, func_b):
        """Return p sum."""
        result = np.zeros_like(func_b[1])
        for p in range(1, 4):
            result += (-1) ** p * func_a[p] * func_b[p]
        return result
