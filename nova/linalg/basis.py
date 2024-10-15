"""Basis function sets for use in regression analysis."""

from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.special
import scipy.sparse
import xarray

from nova.linalg.decompose import Decompose
from nova.graphics.plot import Plot1D


@dataclass
class LinearSample:
    """Generate 1D linear sample."""

    length: int

    def __post_init__(self):
        """Update coordinate."""
        self.update_coordinate(self.length)

    def update_coordinate(self, length=None):
        """Update sample coordinate."""
        if length is not None:
            self.length = length
        self.coordinate = np.linspace(0, 1, self.length)


@dataclass
class Basis(Plot1D, LinearSample):
    """Basis function base class."""

    order: int = 0
    name: ClassVar[str] = "Base"

    def __post_init__(self):
        """Construct interaction matrix and initalize operator."""
        super().__post_init__()
        self.matrix = np.copy(
            np.c_[[self.basis(i) for i in range(self.order + 1)]].T, order="C"
        )

    @property
    def shape(self):
        """Return matrix shape."""
        return self.matrix.shape

    @abstractmethod
    def basis(self, term: int):
        """Return ith term basis."""

    @cached_property
    def function(self):
        """Return basis function for given order."""
        return np.stack([self.basis(i) for i in range(self.order + 1)])

    def plot(self, model=None, **kwargs):
        """Plot set of basis functions evaluated for coordinate."""
        self.axes = kwargs.pop("axes", None)
        if model is None:
            model = np.ones(self.order + 1)
        for i, coef in enumerate(model):
            self.axes.plot(self.coordinate, coef * self.basis(i), **kwargs)
        self.axes.plot(self.coordinate, np.dot(model, self.function), "k")
        self.axes.set_xlabel("coordinate")
        self.axes.set_ylabel("basis")
        self.axes.set_title(self.name)


@dataclass
class Bernstein(Basis):
    """Berstein polynomial regression of a given order."""

    name: ClassVar[str] = "Bernstein Polynomial"

    def basis(self, term: int):
        """Return Bernstein basis polynomial."""
        return (
            scipy.special.binom(self.order, term)
            * self.coordinate**term
            * (1 - self.coordinate) ** (self.order - term)
        )


@dataclass
class SvdAttrs:
    """Non-default attributes for Svd basis class."""

    length: int
    rank: int


@dataclass
class Svd(Basis, SvdAttrs):
    """Construct regression model from Svd reduction."""

    matrix: npt.ArrayLike = None
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    name: ClassVar[str] = "Singular Value Decomposition"

    def __post_init__(self):
        """Update basis order."""
        self.update_coordinate()
        self.order = self.rank - 1

    def __call__(self, data):
        """Append data."""
        self.append_data(data)
        return self

    def __iadd__(self, data):
        """Append data to SVD basis."""
        self.append_data(data)
        return self

    def append_data(self, data):
        """Append data to matrix."""
        data = self._reshape(data)
        if self.matrix is not None:
            data = np.append(self.matrix.T, data, axis=0)
        self.matrix = self._reduce(data)

    def _reshape(self, data):
        """Reshape data to match coordinate length."""
        if (length := data.shape[1]) == self.length:
            return data
        return scipy.interpolate.interp1d(np.linspace(0, 1, length), data, axis=1)(
            self.coordinate
        )

    def _reduce(self, data):
        """Perform svd reduction."""
        decompose = Decompose(data, self.rank)
        return decompose.matrices["V"]

    def interpolate(self, length: int):
        """Interpolate matrix to new coodrinate length."""
        if length == self.length:
            return
        _coordinate = self.coordinate.copy()
        self.update_coordinate(length)
        self.matrix = scipy.interpolate.interp1d(_coordinate, self.matrix, axis=0)(
            self.coordinate
        )

    def basis(self, term: int):
        """Return individual basis."""
        return self.matrix[:, term]


if __name__ == "__main__":
    bernstein = Bernstein(131, 21)
    print(bernstein.shape)
    bernstein.plot()

    """

    eq = EquilibriumData(135011, 7)
    attr = 'f_df_dpsi'

    svd = Svd(50, 5)


    svd += eq.data[attr]

    svd.plot()

    #svd.load_frame('DINA-IMAS', attr)
    #svd.load_frame('CORSICA', attr)

    #eq = EquilibriumData(130506, 403)
    #svd += eq.data[attr]

    svd.plot(ls='--')
    """

    """


    import pylops

    #D2op = pylops.SecondDerivative(bernstein.shape[1],
    #                               dims=None, dtype="float64")

    #bernstein.model = pylops.optimization.leastsquares.RegularizedInversion(
    #    bernstein, [D2op], data,
    #    epsRs=[np.sqrt(1000)], **dict(damp=0, iter_lim=100, show=1)
    #    )

    #Aop = pylops.MatrixMult(bernstein.matrix, dtype="float64")

    #bernstein.model = pylops.optimization.sparsity.FISTA(Aop, data, 5000)[0]
    #bernstein.plot()

    alpha = 0.05
    bernstein = Bernstein(eq.data.sizes['psi_norm'], 21)

    matrix = bernstein.matrix.T @ bernstein.matrix
    matrix.flat[::matrix.shape[1]+1] += alpha
    bernstein.model = np.linalg.lstsq(
        matrix, bernstein.matrix.T @ data)[0]
    bernstein.plot(plt.gca())

    plt.plot(bernstein.coordinate, profile.data)

    #from sklearn import linear_model
    #reg = linear_model.Ridge(alpha=.5)










    """

    """

    import scipy
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.metrics.pairwise import rbf_kernel

    krr = KernelRidge(alpha=0.5, kernel='rbf')

    X = eq.data[attr].data[:10].copy('C')

    y = [bernstein / sample for sample in X]

    krr.fit(X, y)

    bernstein / profile.data
    bernstein.plot()

    bernstein.model = krr.predict(data.reshape(1, -1))[0]
    bernstein.plot(axes=plt.gca())

    plt.plot(bernstein.coordinate, data, 'o')

    bernstein = Bernstein(eq.data.sizes['psi_norm'], 7)
    bernstein /= data
    bernstein.plot(axes=plt.gca())

    """

    '''
    alpha = 0.5
    K = X.copy()

    import numba
    from nova.frame.biotoperate import matmul
    import math

    @numba.njit
    def rbf(X):
        """Compute rbf kernel."""
        n_sample = X.shape[0]
        gamma = 1 / X.shape[1]

        XX = np.sum(X**2, axis=1)
        XY = X @ X.T

        K = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(i+1):
                L2 = XX[i] + XX[j] - 2 * XY[i, j]
                K[i, j] = L2
                #if i == j:
                #    continue
                #K[j, i] = L2
        K = np.exp(-gamma * K)
        return K

    X = X.copy('C')

    K_ = rbf(X)

    K_ = rbf(X)

    K = rbf_kernel(X)
    '''

    """
    ######### svd
    from sklearn.decomposition import TruncatedSVD

    X = eq.data[attr].data[::50].copy('C')
    svd = TruncatedSVD(5)
    X_ = svd.fit_transform(X.T)

    reg = Regression(X_)
    reg.coordinate = bernstein.coordinate
    #reg /= data

    alpha = 1
    matrix = reg.matrix.T @ reg.matrix
    matrix.flat[::matrix.shape[1]+1] += alpha
    reg.model = np.linalg.lstsq(matrix, reg.matrix.T @ data)[0]

    reg.plot(axes=plt.gca())
    ########### svd
    """

    """
    """
    # reg /= data

    """
    bernstein = Bernstein(eq.data.sizes['psi_norm'], 7)

    err = np.zeros((len(eq.data[attr]), eq.data.sizes['psi_norm']))
    for i in range(eq.data.sizes['time']):
        err[i] = eq.data[attr][i] - bernstein.forward(
            bernstein / eq.data[attr].data[i])
    """

    """
    import sklearn.covariance

    cov = sklearn.covariance.OAS().fit(eq.data[attr].data[itime-100:itime])
    cov = cov.covariance_

    cov_ = np.linalg.inv(cov)

    #C = np.linalg.cholesky(cov)
    #C_ = np.linalg.inv(C)

    #data = C_ @ data
    #bernstein.matrix = C_ @ bernstein.matrix[:-1]
    bernstein = Bernstein(eq.data.sizes['psi_norm'], 7)

    alpha = 0
    #matrix = C_ @ bernstein.matrix
    cov_matrix = bernstein.matrix.T @ cov_ @ bernstein.matrix
    cov_matrix.flat[::cov_matrix.shape[1]+1] += alpha
    bernstein.model = np.linalg.lstsq(cov_matrix,
                                      bernstein.matrix.T @ cov_ @ data)[0]
    bernstein.plot(plt.gca())
    #reg.coordinate = bernstein.coordinate
    #reg.plot(axes=plt.gca())
    """

    """
    from sklearn.decomposition import TruncatedSVD


    svd = TruncatedSVD(5)
    X_ = svd.fit_transform(X.T)

    reg = Regression(X_)
    reg /= data

    #reg.model = np.linalg.lstsq(X_, X[0])[0]
    #reg.data = X[0]

    reg.coordinate = np.linspace(0, 1, 50)
    reg.plot(axes=plt.gca())

    plt.plot(reg.coordinate, profile.data)
    """

    """
    reg /= data
    reg.coordinate = bernstein.coordinate
    reg.plot(axes=plt.gca())
    """

    """

    for x in X:
        plt.plot(bernstein.coordinate, x)

    #print(np.allclose(K_, K))
    """

    """
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(3, algorithm='arpack')
    X = svd.fit_transform(X)

    y = [bernstein / sample for sample in X]

    K.flat[:: X.shape[0] + 1] += alpha
    dual_coef = scipy.linalg.solve(K, y, sym_pos=True, overwrite_a=False)

    bernstein.model = rbf_kernel(data.reshape(1, -1), X)[0] @ dual_coef

    # bernstein / data
    bernstein.plot(axes=plt.gca())
    """

    """
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

    #bernstein = Bernstein(eq.data.sizes['psi_norm'], 21)
    #lsq = scipy.optimize.lsq_linear(bernstein.matrix, profile)


    #lop = berstein / profile
    #np.linalg.lstsq(berstein.matrix, profile)

    plt.plot(eq.data.psi_norm, profile)
    #plt.plot(eq.data.psi_norm, bernstein.matrix @ lsq.x, '--')
    plt.plot(eq.data.psi_norm, bernstein(), '-.')
    """
