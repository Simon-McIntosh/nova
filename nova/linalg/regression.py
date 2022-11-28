"""Provide linear operators for regression analysis."""
from abc import abstractmethod
from dataclasses import dataclass, field

import numba
import numpy as np
from pylops import LinearOperator
from pylops.optimization.leastsquares import RegularizedInversion

from nova.imas.equilibrium import Equilibrium
from nova.linalg.decompose import Decompose
from nova.plot.plotter import LinePlot


@dataclass
class RegressionBase(LinePlot):
    """Implement full-matrix forward and inverse models."""

    matrix: np.ndarray = field(repr=False)
    coordinate: np.ndarray | None = field(repr=False, default=None)
    model: np.ndarray | None = field(default=None)
    data: np.ndarray | None = field(repr=False, default=None)

    def __post_init__(self):
        """Update coordinate, model, and data."""
        self.matrix_h = self.matrix.T.copy(order='C')
        self.update_coordinate(self.coordinate)
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

    def update_coordinate(self, coordinate):
        """Update model coordinate, check shape."""
        if coordinate is None:
            self.coordinate = np.linspace(0, 1, len(self.matrix))
            return
        if len(coordinate) != self.shape[0]:
            raise IndexError(f'coordinate length {len(coordinate)} != '
                             f'matrix.shape[0] {self.shape[0]}')
        self.coordinate = coordinate

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
        """Return forward model result."""
        return self.matrix @ self.model

    def forward(self, model=None):
        """Return results of forward model evaluation."""
        self.update_model(model)
        return self._forward()

    def _adjoint(self):
        """Return adjoint operator result."""
        return self.matrix_h @ self.data

    def adjoint(self, data=None):
        """Return results of adjoint transform."""
        self.update_data(data)
        return self._adjoint()

    @staticmethod
    @numba.njit
    def _lstsq(matrix, data):
        """Calcuate inverse via numpy's lstsq method."""
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
    def _lstsq(matrix, data):
        """Calcuate inverse vir numpy's lstsq method."""
        return np.linalg.lstsq(matrix, data)[0]

    def _inverse(self):
        return self._lstsq(self.matrix, self.data)


@dataclass
class Lops(RegressionBase, LinearOperator):
    """Extend Pylops linear operator and Nova regression classes."""

    dtype: type = float
    explicit: bool = True

    def __post_init__(self):
        """Link matrix attribute to Pylops LinearOperator.A."""
        LinearOperator.__init__(self)
        super().__post_init__()
        self.A = self.matrix

    def _matvec(self, model):
        """Return results of forward calculation."""
        return self.forward(model)

    def _rmatvec(self, data):
        """Return results of adjoint calculation."""
        return self.adjoint(data)

    def _inverse(self):
        """Retun solution to least squares problem using default solver."""
        return RegularizedInversion(self).solve(self.data, None)[0]


@dataclass
class MoorePenrose(RegressionBase):
    """Fast operators for linear regression analysis using pseudoinverse."""

    alpha: float = 0
    rank: int = 0

    def __post_init__(self):
        """Perform matrix reduction."""
        super().__post_init__()
        self.matrices = Decompose(self.matrix, self.rank).matrices

    @staticmethod
    def __forward(U, s, Vh, model):
        """Return results of forward model evaluation."""
        return U @ (s * (Vh @ model))

    def _forward(self):
        """Call numba forward model - apply svd reduction if flag==True."""
        self.__forward(self.matrices['U'], self.matrices['s'],
                       self.matrices['Vh'], self.model)

    def __inverse(self, V, s, Uh, data):
        """Calcuate inverse via svd psudo inverse."""
        return V @ ((Uh @ data) * s / (s**2 + self.alpha**2))

    def _inverse(self):
        """Extend Regression._inverse to include option for svd reduction."""
        return self.__inverse(self.matrices['V'], self.matrices['s'],
                              self.matrices['Uh'], self.data)


if __name__ == '__main__':

    from nova.linalg.basis import Svd

    svd = Svd(7, 81)
    attr = 'f_df_dpsi'

    #svd.load_frame('DINA-IMAS', attr)
    svd.load_frame('CORSICA', attr)

    eq = Equilibrium(100504, 3)

    ols = OdinaryLeastSquares(svd.matrix)

    itime = 30
    profile = eq.data[attr][itime]

    rng = np.random.default_rng(2025)
    data = profile.data.copy()
    data += np.std(data) * (rng.random(eq.data.dims['psi_norm']) - 0.5)

    ols /= data

    ols.plot()

    import matplotlib.pyplot as plt
    plt.plot(ols.coordinate, profile.data)


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
