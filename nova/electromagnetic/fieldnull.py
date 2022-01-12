
from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt
import scipy.interpolate

from nova.electromagnetic.biotgrid import BiotGrid
from nova.electromagnetic.coilset import CoilSet
from nova.utilities.pyplot import plt


"""
RectBivariateSpline
Definition : RectBivariateSpline(x, y, z, bbox=[None] * 4, kx=3, ky=3, s=0)

Bivariate spline approximation over a rectangular mesh.

Can be used for both smoothing and interpolating data.

Parameters
x,yarray_like
1-D arrays of coordinates in strictly ascending order.

zarray_like
2-D array of data with shape (x.size,y.size).

bboxarray_like, optional
Sequence of length 4 specifying the boundary of the rectangular approximation domain. By default, bbox=[min(x), max(x), min(y), max(y)].

kx, kyints, optional
Degrees of the bivariate spline. Default is 3.

sfloat, optional
Positive smoothing factor defined for estimation condition: sum((z[i]-f(x[i], y[i]))**2, axis=0) <= s where f is a spline function. Default is s=0, which is for interpolation.
"""


@dataclass
class Interpolator:
    """Implement interface to scipy.RectBivariateSpline."""

    x_coordinate: npt.ArrayLike
    z_coordinate: npt.ArrayLike
    data: npt.ArrayLike
    bounds: list[float] = field(default_factory=lambda: 4*[None])
    degree: int = 2
    smoothing: float = 0

    def __post_init__(self):
        """Generate interpolator."""
        self._spline = scipy.interpolate.RectBivariateSpline(
            self.x_coordinate, self.z_coordinate, self.data,
            self.bounds, self.degree, self.degree, self.smoothing)

    def __call__(self, *args):
        """Return spline evaulation."""
        return self._spline.ev(*args)


@dataclass
class EvGrid:

    grid: BiotGrid
    attrs: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize attribute IDs"""
        self.attrs = {attr: id(None) for attr in self.grid.attrs}

#import hdbscan
import numba


@dataclass
class FieldGrid:

    x_coordinate: npt.ArrayLike
    z_coordinate: npt.ArrayLike
    x_dim: int = field(init=False)
    z_dim: int = field(init=False)
    points: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        """Construct grid from input coordinates."""
        self.xdim = len(self.x_coordinate)
        self.zdim = len(self.z_coordinate)
        x2d, z2d = np.meshgrid(self.x_coordinate, self.z_coordinate,
                               indexing='ij')
        self.points = np.c_[x2d.flatten(), z2d.flatten()]


@numba.stencil(func_or_mode='constant', cval=-1,
               neighborhood=((-1, 1), (-1, 1)))
def counter(data):
    center = data[0, 0]
    sign = data[-1, 1] > center
    count = 0
    for k in [(-1, 0), (-1, -1), (0, -1), (1, -1),
              (1, 0), (1, 1), (0, 1), (-1, 1)]:
        _sign = data[k] > center
        if _sign != sign:
            count += 1
            sign = _sign
    return count

'''
@numba.experimental.jitclass(
    dict(field=numba.float64[:, :], flux=numba.float64[:, :],
         index_x=numba.boolean[:, :], index_o=numba.boolean[:, :],
         pattern=numba.int16[:, :]))
'''
class FieldNull:

    def __init__(self, field, flux):
        self.field = field
        self.flux = flux

        #self.pattern = np.array([[-1, 0], [-1, -1], [0, -1], [1, -1],
        #                         [1, 0], [1, 1], [0, 1], [-1, 1]],
        #                        dtype=numba.int16)

        #self.index_o, self.index_x
        #field_count = categorize(self.field)
        #flux_count = categorize(self.flux)

        #self.index_o = self.minimum_search(self.flux)
        #self.index_o |= self.minimum_search(-self.flux)
        #self.index_x = self.minimum_search(self.field)
        #self.index_x &= ~self.index_o

        #self.index_o = flux_count == 0
        #self.index_x = (flux_count == 4) & (field_count == 0)

        self.index_o, self.index_x = self.categorize(self.flux)
        self.index_x &= self.categorize(self.field)[0]

        #count = counter(self.flux)
        #self.index_o = count == 0
        #self.index_x = count == 4

    @staticmethod
    def minimum_search(data):
        xdim, zdim = data.shape
        index = np.full((xdim, zdim), False)
        for i in numba.prange(1, xdim-1):
            for j in numba.prange(1, zdim-1):
                if data[i-1, j] < data[i, j]:
                    continue
                if data[i+1, j] < data[i, j]:
                    continue
                if data[i, j-1] < data[i, j]:
                    continue
                if data[i, j+1] < data[i, j]:
                    continue
                index[i, j] = True
        return index

    def pindex(self, i, j, k):
        index = self.pattern[k]
        return i+index[0], j+index[1]

    @staticmethod
    @numba.njit()
    def categorize(data):
        """Categorize points in 2D grid.

        Count number of sign changes whilst traversing neighbour point loop.

            - 0: minima / maxima point
            - 2: regular point
            - 4: saddle point

        """
        xdim, zdim = data.shape
        index_o = np.full((xdim, zdim), False, dtype=numba.boolean)
        index_x = np.full((xdim, zdim), False, dtype=numba.boolean)
        for i in numba.prange(1, xdim-1):
            for j in range(1, zdim-1):
                center = data[i, j]
                sign = data[i-1, j+1] > center
                count = 0
                for k in [(-1, 0), (-1, -1), (0, -1), (1, -1),
                          (1, 0), (1, 1), (0, 1), (-1, 1)]:
                    _sign = data[i+k[0], j+k[1]] > center
                    if _sign != sign:
                        count += 1
                        sign = _sign
                if count == 0:
                    index_o[i, j] = True
                if count == 4:
                    index_x[i, j] = True
        return index_o, index_x

    @staticmethod
    def filter2d(image, filt):
        """Return filtered data.

        http://numba.pydata.org/numba-doc/0.15.1/examples.html
        """
        M, N = image.shape
        Mf, Nf = filt.shape
        Mf2 = Mf // 2
        Nf2 = Nf // 2
        result = np.zeros_like(image)
        for i in range(Mf2, M - Mf2):
            for j in range(Nf2, N - Nf2):
                num = 0.0
                for ii in range(Mf):
                    for jj in range(Nf):
                        num += (filt[Mf-1-ii, Nf-1-jj] *
                                image[i-Mf2+ii, j-Nf2+jj])
                result[i, j] = num
        return result
    '''
    @staticmethod
    def kernel(sigma=1, length=3):
        """
        Return normalized Gaussian kernel.

        https://stackoverflow.com/questions/29731726/
        how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        """
        ax = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)
    '''

if __name__ == '__main__':

    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.plasma.insert(dict(o=(4, 0, 0.5)), delta=0.3)
    coilset.grid.solve(1000, 0.05) #[3.2, 8.5, -2.5, 2.5])
    coilset.sloc['Ic'] = -15e6

    grid = coilset.grid

    shape = grid.data.dims['x'], grid.data.dims['z']

    #bn = scipy.ndimage.gaussian_filter(grid.bn.reshape(shape), 1.5)

    null = Null(grid.bn.reshape(shape), grid.psi.reshape(shape))

    plt.plot(grid.data.x2d.data[null.index_o],
             grid.data.z2d.data[null.index_o], 'o')
    plt.plot(grid.data.x2d.data[null.index_x],
             grid.data.z2d.data[null.index_x], 'X')


    coilset.plot()
    coilset.grid.plot()

'''
@dataclass
class FieldNull:
    """Locate field null clusters based on a lower quantile search."""

    x_coordinate: npt.ArrayLike
    z_coordinate: npt.ArrayLike
    field: npt.ArrayLike
    flux: npt.ArrayLike
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        """Reshape input (1D to 2D)"""
        self.shape = len(self.x_coordinate), len(self.z_coordinate)
        self.field.shape = self.shape
        self.flux.shape = self.shape

        self.topology(self.field, self.flux)
        #field_null = self.search(self.field)
        #flux_null = self.search(self.flux) or self.search(self.flux, False)

    @staticmethod
    @numba.njit(parallel=True)
    def topology(field, flux):
        field_null = FieldNull.search(field)
        #flux_null = FieldNull.search(flux) & FieldNull.search(flux, False)

    @staticmethod
    @numba.njit(parallel=True)
    def search(data, minimize=True):
        xdim, zdim = data.shape
        index = np.full((xdim, zdim), False)
        for i in numba.prange(1, xdim-1):
            for j in numba.prange(1, zdim-1):
                if data[i-1, j] < minimize*data[i, j]:
                    continue
                if data[i+1, j] < minimize*data[i, j]:
                    continue
                if data[i, j-1] < minimize*data[i, j]:
                    continue
                if data[i, j+1] < minimize*data[i, j]:
                    continue
                index[i, j] = True
        return index


    """
    # field null clusters
    self._null_cluster, _field_Opoint = [], []
    if self.cluster:  # cluster low field nulls
        Bthreshold = np.quantile(self.B, field_quantile,
                                 interpolation='lower')
        index = self.B < Bthreshold  # threshold
        if np.sum(index) > 0:  # protect against uniform zero field
            xt, zt = self.x2d[index], self.z2d[index]  # threshold points
            dbscan = sklearn.cluster.DBSCAN(eps=eps_cluster, min_samples=1)
            cluster_index = dbscan.fit_predict(np.array([xt, zt]).T)
            for i in range(np.max(cluster_index)+1):
                # cluster coordinates
                x_cluster = xt[cluster_index == i]
                z_cluster = zt[cluster_index == i]
                self._null_cluster.append([x_cluster, z_cluster])
    """


if __name__ == '__main__':

    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.plasma.insert(dict(o=(4, 0, 0.5)), delta=0.3)
    coilset.grid.solve(400, [3.2, 8.5, -2.5, 2.5])
    coilset.sloc['Ic'] = -15e6

    grid = coilset.grid

    B2d = grid.bn.reshape((grid.data.dims['x'], grid.data.dims['z']))
    Psi2d = grid.psi.reshape((grid.data.dims['x'], grid.data.dims['z']))
    scipy.interpolate.RectBivariateSpline(
        grid.data['x'], grid.data['z'], B2d)

    null = FieldNull(grid.data['x'], grid.data['z'],
                     grid.bn, grid.psi)

    index = null.search(Psi2d)

    plt.plot(grid.data.x2d.data[index], grid.data.z2d.data[index], 'o')

    coilset.plot()
    coilset.grid.plot(levels=81)
'''
