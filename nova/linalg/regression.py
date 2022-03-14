"""Manage Berstein polynomials."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.special
import scipy.sparse

from nova.electromagnetic.biotoperate import matmul
from nova.imas.equilibrium import Equilibrium
from nova.linalg.decompose import Decompose
from nova.utilities.pyplot import plt


@dataclass
class Basis(ABC):
    """Basis function base class."""

    length: int
    order: int

    def __post_init__(self):
        """Construct interaction matrix and initalize operator."""
        self.coordinate = np.linspace(0, 1, self.length)
        self.matrix = np.c_[[self.basis(i) for i in range(self.order+1)]].T
        self.shape = self.matrix.shape

    @abstractmethod
    def basis(self, term: int):
        """Return ith term basis."""

    def plot(self, model=None, **kwargs):
        """Plot set of basis functions evaluated for coordinate."""
        if model is None:
            model = np.ones(self.order+1)
        for i, coef in enumerate(model):
            plt.plot(self.coordinate, coef * self.basis(i), **kwargs)


@dataclass
class Regression(Basis):
    """Implement full-matrix forward and inverse models."""

    model: npt.ArrayLike = None
    data: npt.ArrayLike = None

    def __call__(self):
        """Evaluate model."""
        return self.forward()

    def __truediv__(self, data):
        """Solve inverse model given data."""
        return self.inverse(data)

    def __itruediv__(self, data):
        """Solve inverse model inplace."""
        self.__truediv__(data)
        return self

    @staticmethod
    @numba.njit
    def __forward(matrix, model):
        """Class local numba forward evaluation."""
        return matmul(matrix, model)

    def _forward(self):
        """Call numba forward model."""
        return self.__forward(self.matrix, self.model)

    def update_model(self, model):
        """Update model coefficents, check shape."""
        if model is not None:
            if len(model) != self.shape[1]:
                raise IndexError(f'model length {len(self.model)} != '
                                 f'matrix.shape[1] {self.shape[1]}')
            self.model = model
        if self.model is None:
            raise AttributeError('model not set, '
                                 'unable to solve forward model')

    def forward(self, model=None):
        """Return results of forward model evaluation."""
        self.update_model(model)
        return self._forward()

    @staticmethod
    @numba.njit
    def __inverse(matrix, data):
        """Calcuate inverse vir numpy's lstsq method."""
        return np.linalg.lstsq(matrix, data)[0]

    def _inverse(self):
        return self.__inverse(self.matrix, self.data)

    def update_data(self, data):
        """Update data, check length."""
        if len(data) != self.shape[0]:
            raise IndexError(f'len(data) {len(data)} != '
                             f'matrix.shape[0] {self.shape[0]}')
        if not isinstance(data, np.ndarray):
            raise TypeError(f'type(data) {type(data)} is not ndarray')
        self.data = data

    def inverse(self, data):
        """Calculate inverse and update model coeffcents."""
        self.update_data(data)
        self.model = self._inverse()
        return self.model

    def plot(self):
        """Plot fit."""
        axes = plt.subplots(1, 1)[1]
        if self.data is not None:
            axes.plot(self.coordinate, self.data, label='data')
        if self.model is not None:
            # super().plot(self.model, color='gray', alpha=0.5)
            axes.plot(self.coordinate, self.forward(), '--', label='fit')
        plt.despine()
        axes.legend()


@dataclass
class RegressionSvd(Decompose, Regression):
    """Fast operators for linear regression analysis."""

    @staticmethod
    @numba.njit
    def __forward(U, s, Vh, model):
        """Return results of forward model evaluation."""
        return matmul(U, s * matmul(Vh, model))

    def _forward(self):
        """Call numba forward model - apply svd reduction if flag==True."""
        if self.svd:
            return self.__forward(self._svd['U'], self._svd['s'],
                                  self._svd['Vh'], self.model)
        return super()._forward()

    @staticmethod
    @numba.njit
    def __inverse(V, s, Uh, data):
        """Calcuate inverse via svd psudo inverse."""
        return matmul(V, matmul(Uh, data) / s)

    def _inverse(self):
        """Extend Regression._inverse to include option for svd reduction."""
        if self.svd:
            return self.__inverse(self._svd['V'], self._svd['s'],
                                  self._svd['Uh'], self.data)
        return super()._inverse()


@dataclass
class Bernstein(RegressionSvd):
    """Berstein polynomial regression of a given order."""

    def basis(self, term: int):
        """Return Bernstein basis polynomial."""
        super().basis(term)
        return scipy.special.binom(self.order, term) * \
            self.coordinate**term * (1 - self.coordinate)**(self.order - term)


if __name__ == '__main__':

    #berstein.plot()

    eq = Equilibrium(135011, 7)
    # eq.build()
    attr = 'f_df_dpsi'
    #attr = 'dpressure_dpsi'
    itime = 200
    profile = eq.data[attr][itime]

    bernstein = Bernstein(eq.data.dims['psi_norm'], 21, svd=False)

    bernstein /= profile.data

    bernstein.plot()

    '''

    #bernstein = Bernstein(eq.data.dims['psi_norm'], 21)
    #lsq = scipy.optimize.lsq_linear(bernstein.matrix, profile)


    #lop = berstein / profile
    #np.linalg.lstsq(berstein.matrix, profile)

    plt.plot(eq.data.psi_norm, profile)
    #plt.plot(eq.data.psi_norm, bernstein.matrix @ lsq.x, '--')
    plt.plot(eq.data.psi_norm, bernstein(), '-.')
    '''
