
from dataclasses import dataclass

import numpy as np
import numba
from numba.experimental import jitclass
import scipy.special


def gamma(zs, z):
    """Return gamma coefficient."""
    return zs - z


def a2(rs, r, zs, z):
    """Return a**2 coefficient."""
    return gamma(zs, z)**2 + (rs + r)**2


def b(rs, r):
    """Return b coefficient."""
    return rs + r


def c2(r, zs, z):
    """Return c**2 coefficient."""
    return gamma(zs, z)**2 + r**2


def c(r, zs, z):
    """Return c coefficient."""
    return np.sqrt(c2(r, zs, z))


def k2(rs, r, zs, z):
    """Return k2 coefficient."""
    return 4*r*rs / a2(rs, r, zs, z)


def np2(rs, r, zs, z, p):
    """Return np**2 constants."""
    if p == 1:
        return 2*r / (r - c(r, zs, z))
    if p == 2:
        return 2*r / (r + c(r, zs, z))
    if p == 3:
        return 4*r*rs / b(rs, r)**2
    return np.zeros_like(r)


def Pphi(rs, r, zs, z, p):
    """Return Pphi coefficient, q in [1, 2, 3]."""
    if p == 3:
        return -rs / b(rs, r) * (rs - r) * (3*r**2 - rs**2)
    _c = c(r, zs, z)
    _np2 = np2(rs, r, zs, z, p)
    return (rs - (-1)**p * _c) * _np2 * _c * (3*r**2 - _c**2) / (2*r)


def U(rs, r, zs, z):
    """Return U coefficient."""
    return k2(rs, r, zs, z) * (4*gamma(zs, z)**2 + 3*rs**2 - 5*r**2) / (4*r)


def p_sum(rs, r, zs, z, method):
    """Return p sum."""
    result = np.zeros_like(rs)
    for p in range(1, 4):
        result += (-1)**p * method(rs, r, zs, z, p)
    #    result += (-1)**p * np2(rs, r, zs, z, p, eps)  #* self.Pi[p]
    return result


numba.jit_module(nopython=True, cache=True, parallel=True)


class Ellip:

    def __init__(self, rs, r, zs, z):
        """Compute elliptic intergrals of the 1st, 2nd and 3rd kind."""
        _k2 = k2(rs, r, zs, z)
        self.K = self._ellip('k', _k2)
        self.E = self._ellip('e', _k2)
        self.Pi = {p: self.ellippi(np2(rs, r, zs, z, p), _k2)
                   for p in range(1, 4)}

    @staticmethod
    def _ellip(kind: str, /, *args, out=None, shape=None, where=True):
        if out is None:
            out = np.zeros_like(args[0], dtype=float, shape=shape)
        func = getattr(scipy.special, f'ellip{kind}')
        return func(*args, out=out, where=where)

    @classmethod
    def ellippi(cls, n, m):
        """
        Return complete elliptic intergral of the 3rd kind.

        Adapted from https://github.com/scipy/scipy/issues/4452.
        """
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = cls._ellip('rf', x, y, z, shape=m.shape)
        rj = cls._ellip('rj', x, y, z, p, shape=m.shape)
        return rf + rj * n / 3


if __name__ == '__main__':


    rng = np.random.default_rng(2025)

    rs = rng.random((10000, 2000))
    r = rng.random((10000, 2000))
    zs = rng.random((10000, 2000))
    z = rng.random((10000, 2000))

    ellip = Ellip(r, z, rs, zs)

    print(p_sum(rs, r, zs, z, np2))

    #print(Ellip(r, z, rs, zs).K())
    #biot = Biot(k2)
    #print(biot.elip())
