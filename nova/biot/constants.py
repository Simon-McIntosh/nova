"""Biot-Savart intergration constants."""
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import ClassVar

import dask.array as da
import numpy as np
import scipy.special

Array = da.Array | np.ndarray

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
    if threshold_factor is not None and limit_factor is not None and \
            limit_factor > threshold_factor:
        raise ValueError('limit_factor > threshold_factor '
                         f'{limit_factor} > {threshold_factor}')

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            output = method(self, *args, **kwargs)

            def defactor(factor, eps, default):
                if factor is None:
                    return default
                return factor*eps

            limit = defactor(limit_factor, self.eps, 0)
            threshold = defactor(threshold_factor, self.eps, 1)
            delta = output - (1 - threshold)
            unit_delta = delta / threshold
            return np.where((output < 1 + limit) & (delta > 0),
                            (1 - threshold) + unit_delta*(threshold - limit),
                            output)
        return wrapper
    return decorator


@dataclass
class Constants:
    """Manage biot intergration constants."""

    rs: Array = field(default_factory=lambda: da.zeros_like([]))
    zs: Array = field(default_factory=lambda: da.zeros_like([]))
    r: Array = field(default_factory=lambda: da.zeros_like([]))
    z: Array = field(default_factory=lambda: da.zeros_like([]))

    eps: ClassVar[np.float64] = 2*np.finfo(float).eps

    def sign(self, x):
        """Return sign of array -1 if x < 0 else 1."""
        return np.where(abs(x) > 1e4*self.eps, np.sign(x), 0)

    @cached_property
    def gamma(self):
        """Return gamma coefficient."""
        return self.zs - self.z

    @property
    def a2(self):
        """Return a**2 coefficient."""
        return self.gamma**2 + (self.rs + self.r)**2

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
        return (1-self.eps) * 4*self.r*self.rs / self.a2

    @property
    def ck2(self):
        """Return complementary modulus."""
        return 1 - self.k2

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
        return self.k2 * (4*self.gamma**2 +
                          3*self.rs**2 - 5*self.r**2) / (4*self.r)

    @staticmethod
    def _ellip(kind: str, /, *args, out=None, shape=None, where=True):
        if out is None:
            out = np.zeros_like(args[0], dtype=float, shape=shape)
        func = getattr(scipy.special, f'ellip{kind}')
        return func(*args, out=out, where=where)

    @classmethod
    def ellipk(cls, m):
        """Return complete elliptic intergral of the 1st kind."""
        #return scipy.special.ellipk(m)
        return cls._ellip('k', m)

    @classmethod
    def ellipe(cls, m):
        """Return complete elliptic intergral of the 2nd kind."""
        #return scipy.special.ellipe(m)
        return cls._ellip('e', m)

    @classmethod
    def ellippi(cls, n, m):
        """
        Return complete elliptic intergral of the 3rd kind.

        Adapted from https://github.com/scipy/scipy/issues/4452.
        """
        '''
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = cls._ellip('rf', x, y, z, shape=m.shape) #, where=(m < 1))
        #cls._ellip('rc', 0, 1, out=rf, where=np.isclose(y, 1))
        rj = cls._ellip('rj', x, y, z, p, shape=m.shape) #, where=(m < 1))
        #cls._ellip('rd', x, z, p, out=rj, where=np.isclose(y, p))
        #cls._ellip('rd', x, y, p, out=rj, where=np.isclose(p, 1))
        return rf + rj * n / 3
        '''
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = scipy.special.elliprf(x, y, z)
        rf[(y == 0) | (y == 1)] = scipy.special.elliprc(0, 1)
        rj = scipy.special.elliprj(x, y, z, p)
        rj[y == p] = scipy.special.elliprd(x, z, p[y == p])
        rj[p == 1] = scipy.special.elliprd(x, y[p == 1], 1)
        return rf + rj * n / 3


    #@unit_nudge()
    def _np2_2(self):
        return 2*self.r / (self.r + self.c)

    #@unit_nudge()
    def _np2_3(self):
        return 4*self.r*self.rs / self.b**2

    @cached_property
    def np2(self) -> dict[int, np.ndarray]:
        """Return np**2 constant."""
        return {1: 2*self.r / (self.r - self.c - self.eps),
                2: (1-self.eps) * 2*self.r / (self.r + self.c),
                3: (1-self.eps) * 4*self.r*self.rs / self.b**2}

    @cached_property
    def Pphi(self) -> dict[int, np.ndarray]:
        """Return Pphi coefficient, q in [1, 2, 3]."""
        Pphi = {p: (self.rs - (-1)**p * self.c) * self.np2[p] * self.c *
                (3*self.r**2 - self.c2) / (2*self.r) for p in [1, 2]}
        Pphi[3] = -self.rs / self.b * (self.rs - self.r) * \
            (3*self.r**2 - self.rs**2)
        return Pphi

    @cached_property
    def Pi(self) -> dict[int, np.ndarray]:
        """Return complete elliptc intergral of the 3rd kind."""
        return {p: self.ellippi(self.np2[p], self.k2) for p in range(1, 4)}

    def p_sum(self, func):
        """Return p sum."""
        result = np.zeros_like(self.r)
        for p in range(1, 4):
            result += (-1)**p * func[p] * self.Pi[p]
        return result
