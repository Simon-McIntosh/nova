
from dataclasses import dataclass

import scipy.interpolate

from nova.electromagnetic.biotgrid import BiotGrid


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
class BaseInterpolate:


@dataclass
class Interpolate:

    grid: BiotGrid
    attrs: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize attribute IDs"""
        self.attrs = {attr: id(None) for attr in self.grid.attrs}

        scipy.interpolate.RectBivariateSpline(
            self.x, self.z, z, bbox=self.grid_boundary))
