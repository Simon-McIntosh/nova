"""Basis function sets for use in regression analysis."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.special
import scipy.sparse

from nova.imas.equilibrium import Equilibrium
from nova.linalg.regression import Regression
from nova.utilities.pyplot import plt


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
class Basis(BasisFunction, Regression, BasisAttributes):
    """Combination class of basis functions and linear operators."""


@dataclass
class Bernstein(Basis):
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
    itime = 500
    profile = eq.data[attr][itime]

    data = profile.data.copy()
    data += np.random.random(eq.data.dims['psi_norm'])-0.5

    bernstein = Bernstein(eq.data.dims['psi_norm'], 25)

    from sklearn.kernel_ridge import KernelRidge

    krr = KernelRidge(alpha=10, kernel='linear')

    X = eq.data[attr].data

    y = [bernstein / sample for sample in X]

    krr.fit(X, y)

    bernstein / profile.data
    bernstein.plot()

    bernstein.model = krr.predict(data.reshape(1, -1))[0]
    bernstein.plot(axes=plt.gca())

    plt.plot(bernstein.coordinate, data, 'o')

    #bernstein / data
    #bernstein.plot(axes=plt.gca())



    '''
    #cov = np.cov(eq.data[attr][:, :-1].T)
    #cov = graphical_lasso(cov, 0.01)
    #cov = GraphicalLassoCV().fit(eq.data[attr][:, :-1]).covariance_

    cov = ledoit_wolf(eq.data[attr][:, :-1])[0]

    C = np.linalg.cholesky(cov)
    C_ = np.linalg.inv(C)

    bernstein.coordinate = bernstein.coordinate[:-1]
    data = C_ @ data[:-1]
    bernstein.matrix = C_ @ bernstein.matrix[:-1]

    bernstein /= data

    bernstein.matrix = C @ bernstein.matrix
    bernstein.data = C @ data

    bernstein.plot()

    #plt.plot(bernstein.coordinate, data_[:-1])
    plt.plot(bernstein.coordinate, profile.data[:-1])

    #bernstein = Bernstein(eq.data.dims['psi_norm'], 21)
    #lsq = scipy.optimize.lsq_linear(bernstein.matrix, profile)


    #lop = berstein / profile
    #np.linalg.lstsq(berstein.matrix, profile)

    plt.plot(eq.data.psi_norm, profile)
    #plt.plot(eq.data.psi_norm, bernstein.matrix @ lsq.x, '--')
    plt.plot(eq.data.psi_norm, bernstein(), '-.')
    '''
