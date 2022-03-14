"""Provide linear operators for regression analysis."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator
import scipy.optimize
import scipy.special
import scipy.sparse

from nova.electromagnetic.biotoperate import matmul
from nova.imas.equilibrium import Equilibrium
from nova.linalg.decompose import Decompose
from nova.utilities.pyplot import plt


@dataclass
class Regression:
    """Implement full-matrix forward and inverse models."""

    matrix: npt.ArrayLike = field(repr=False, default=None)
    model: npt.ArrayLike = field(default=None)
    data: npt.ArrayLike = field(repr=False, default=None)
    matrix_H: npt.ArrayLike = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Calculate adjoint and update model and data."""
        self.matrix_H = self.matrix.T.copy(order='C')
        if self.model is not None:
            self.update_model(self.model)
        if self.data is not None:
            self.update_data(self.data)

    @property
    def shape(self):
        """Return matrix shape."""
        return self.matrix.shape

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

    def update_model(self, model):
        """Update model coefficents, check shape."""
        if model is None:
            if self.model is not None:
                return
            raise AttributeError('model not set, '
                                 'unable to solve forward model')
        if len(model) != self.shape[1]:
            raise IndexError(f'model length {len(model)} != '
                             f'matrix.shape[1] {self.shape[1]}')
        self.model = model

    def _forward(self):
        """Call numba matmul."""
        return matmul(self.matrix, self.model)

    def forward(self, model=None):
        """Return results of forward model evaluation."""
        self.update_model(model)
        return self._forward()

    def _adjoint(self):
        """Call numba matmul operator."""
        return matmul(self.matrix_H, self.data)

    def adjoint(self, data=None):
        """Return results of adjoint transform."""
        self.update_data(data)
        return self._adjoint()

    @staticmethod
    @numba.njit
    def _lstsq(matrix, data):
        """Calcuate inverse vir numpy's lstsq method."""
        return np.linalg.lstsq(matrix, data)[0]

    def _inverse(self):
        return self._lstsq(self.matrix, self.data)

    def update_data(self, data):
        """Update data if not None, check length."""
        if data is None:
            if self.data is not None:
                return
            raise AttributeError('data not set, '
                                 'unable to solve inverse problem')
        if len(data) != self.shape[0]:
            raise IndexError(f'len(data) {len(data)} != '
                             f'matrix.shape[0] {self.shape[0]}')
        if not isinstance(data, np.ndarray):
            raise TypeError(f'type(data) {type(data)} is not ndarray')
        self.data = data

    def inverse(self, data=None):
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
class RegressionLops(LinearOperator, Regression):
    """Extend Pylops linear operator and Nova regression classes."""

    dtype: type = float
    explicit: bool = True

    def __post_init__(self):
        """Link matrix attribute to Pylops LinearOperator.A."""
        super().__post_init__()
        self.A = self.matrix

    def _matvec(self, model):
        """Return results of forward calculation."""
        return matmul(self.matrix, model)

    def _rmatvec(self, data):
        """Return results of adjoint calculation."""
        return matmul(self.matrix_H, data)


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
            return self.__forward(self.matrices['U'], self.matrices['s'],
                                  self.matrices['Vh'], self.model)
        return super()._forward()

    @staticmethod
    @numba.njit
    def __inverse(V, s, Uh, data):
        """Calcuate inverse via svd psudo inverse."""
        return matmul(V, matmul(Uh, data) / s)

    def _inverse(self):
        """Extend Regression._inverse to include option for svd reduction."""
        if self.svd:
            return self.__inverse(self.matrices['V'], self.matrices['s'],
                                  self.matrices['Uh'], self.data)
        return super()._inverse()


@dataclass
class BasisFunction(ABC):
    """Basis function base class."""

    length: int
    order: int

    def __post_init__(self):
        """Construct interaction matrix and initalize operator."""
        self.coordinate = np.linspace(0, 1, self.length)
        self.matrix = np.copy(
            np.c_[[self.basis(i) for i in range(self.order+1)]].T, order='C')
        super().__post_init__()

    @abstractmethod
    def basis(self, term: int):
        """Return ith term basis."""

    def plot_basis(self, model=None, **kwargs):
        """Plot set of basis functions evaluated for coordinate."""
        if model is None:
            model = np.ones(self.order+1)
        for i, coef in enumerate(model):
            plt.plot(self.coordinate, coef * self.basis(i), **kwargs)


@dataclass
class BasisAttributes:
    """Basis function non-default attributes."""

    length: int
    order: int


@dataclass
class Bernstein(BasisFunction, Regression, BasisAttributes):
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

    bernstein = Bernstein(eq.data.dims['psi_norm'], 21)

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
