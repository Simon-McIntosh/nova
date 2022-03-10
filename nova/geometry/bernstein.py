"""Manage Berstein polynomials."""
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.special
import scipy.sparse
import xarray

from nova.electromagnetic.biotoperate import matmul
from nova.imas.equilibrium import Equilibrium
from nova.utilities.pyplot import plt


@numba.experimental.jitclass(dict(
    coordinate=numba.float64[::1],
    matrix=numba.float64[:, ::1],
    svd_rank=numba.int32,
    _U=numba.float64[:, ::1],
    _s=numba.float64[::1],
    _V=numba.float64[:, ::1]))
class Op:
    """Fast linear operators."""

    def __init__(self, coordinate, matrix, svd_rank=-1):
        self.coordinate = coordinate
        self.matrix = matrix
        #  perform svd order reduction
        _U, _s, _V = np.linalg.svd(self.matri)
        self.svd_rank = min([len(_s), svd_rank])
        self._U = _U[:, :self.svd_rank].copy()
        self._s = _s[:self.svd_rank].copy()
        self._V = _V[:self.svd_rank, :].copy()
        assert all(self._s > 0)

    def __call__(self):
        """Return interaction."""
        if self.svd_rank == -1:
            return matmul(self.matrix, self.coordinate)

    def __div__(self, data):
        """Calculate inverse."""
        if self.svd_rank == -1:
            return np.linalg.lstsq(self.matrix, data)
        return matmul(self._V, 1/self._s * matmul(self._U.T, data))


@dataclass
class BernsteinRegression:
    """Berstein polynomial regression of a given order."""

    length: int
    order: int
    coordinate: npt.ArrayLike = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False)
    matrix: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        """Generate linear spaced coordinate array from length attribute."""
        self.coordinate = np.linspace(0, 1, self.length)
        self.matrix = np.c_[[self.basis(i) for i in range(self.order+1)]].T

    def basis(self, term: int):
        """Return Bernstein basis polynomial, 0<=term<degree."""
        assert term >= 0 & term <= self.order
        return scipy.special.binom(self.order, term) * \
            self.coordinate**term * (1 - self.coordinate)**(self.order - term)

    def plot(self):
        """Plot set of Berstein basis polynomials."""
        for i in range(self.order+1):
            plt.plot(self.coordinate, self.basis(i))


if __name__ == '__main__':

    #berstein.plot()

    eq = Equilibrium(135011, 7)
    attr = 'f_df_dpsi'
    itime = 500
    profile = eq.data[attr][itime]

    berstein = BernsteinRegression(eq.data.dims['psi_norm'], 21)

    #np.array.shape

    lsq = scipy.optimize.lsq_linear(berstein.matrix, profile)
    #lop = berstein / profile
    #np.linalg.lstsq(berstein.matrix, profile)

    plt.plot(eq.data.psi_norm, profile)
    plt.plot(eq.data.psi_norm, berstein.matrix @ lsq.x, '--')
