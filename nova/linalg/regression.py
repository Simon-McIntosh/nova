"""Provide linear operators for regression analysis."""
from abc import abstractmethod
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator

from nova.electromagnetic.biotoperate import matmul
from nova.imas.equilibrium import Equilibrium
from nova.utilities.plotter import Line
from nova.linalg.basis import Basis, Svd, Bernstein


@dataclass
class RegressionBase(Line):
    """Implement full-matrix forward and inverse models."""

    basis: Basis
    model: npt.ArrayLike = field(default=None)
    data: npt.ArrayLike = field(repr=False, default=None)

    def __post_init__(self):
        """Calculate adjoint and update model and data."""
        if self.model is not None:
            self.update_model(self.model)
        if self.data is not None:
            self.update_data(self.data)

    @property
    def matrix(self):
        """Return basis matrix."""
        return self.basis.matrix

    @property
    def matrix_H(self):
        """Return matrix transpose."""
        return self.matrix.T.copy(order='C')

    @property
    def coordinate(self):
        """Return basis coordinate."""
        return self.basis.coordinate

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

    @abstractmethod
    def _inverse(self):
        """Solve inverse problem."""

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

    def plot(self, axes=None):
        """Plot fit."""
        self.axes = axes
        if self.data is not None:
            self.axes.plot(self.coordinate, self.data, label='data')
        if self.model is not None:
            self.axes.plot(self.coordinate, self.forward(), '--', label='fit')
        if self.data is not None or self.model is not None:
            self.axes.legend()


@dataclass
class OdinaryLeastSquares(RegressionBase):
    """Implement full-matrix forward and inverse models."""

    @staticmethod
    @numba.njit
    def _lstsq(matrix, data):
        """Calcuate inverse vir numpy's lstsq method."""
        return np.linalg.lstsq(matrix, data)[0]

    def _inverse(self):
        return self._lstsq(self.matrix, self.data)


if __name__ == '__main__':

    basis = Svd(5, 50)
    attr = 'f_df_dpsi'

    eq = Equilibrium(135010, 5)
    basis += eq.data[attr]

    eq = Equilibrium(135011, 7)
    #basis += eq.data[attr]

    eq = Equilibrium(130506, 403)
    basis += eq.data[attr]

    #basis.interpolate(81)
    # basis = Bernstein(50, 9)

    ols = OdinaryLeastSquares(basis)

    eq = Equilibrium(135011, 7)

    ols /= eq.data[attr][100].data

    ols.plot()
'''
@dataclass
class Lops(RegressionBase, LinearOperator):
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
class Svd(RegressionBase, Decompose):
    """Fast operators for linear regression analysis."""

    svd: bool = True

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
'''

if __name__ == '__main__':


    '''

    #berstein.plot()

    eq = Equilibrium(135011, 7)
    # eq.build()
    attr = 'f_df_dpsi'
    #attr = 'dpressure_dpsi'
    itime = 500
    profile = eq.data[attr][itime]

    rng = np.random.default_rng(2025)

    data = profile.data.copy()
    data += 2 * np.std(data) * (rng.random(eq.data.dims['psi_norm']) - 0.5)

    bernstein = Bernstein(eq.data.dims['psi_norm'], 21)
    bernstein /= data
    bernstein.plot()
    '''
