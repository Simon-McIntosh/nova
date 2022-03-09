# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:03:47 2022

@author: mcintos
"""


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
