
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt

from nova.electromagnetic.coilset import CoilSet
from nova.utilities.pyplot import plt


class DataNull:

    mask: npt.ArrayLike
    x_index: npt.ArrayLike = field(init=False)
    z_index: npt.ArrayLike = field(init=False)

    def __post_init__(self):
        x_index, z_index = np.where(self.mask)
        self.index = [(i, j) for i, j in zip(*np.where(self.mask))]
        self.number

    def sort:

    def delete:





@dataclass
class FieldNull:

    r_coordinate: npt.ArrayLike
    z_coordinate: npt.ArrayLike

    def __post_init__(self):
        """Store first-wall profile."""

    def update(self, flux, bnorm=None):
        """Calculate field nulls."""
        x_mask, o_mask = self.categorize(flux)
        if field is not None:
            x_mask &= self.minimum(bnorm)


        self.x_point = sum(self.index_x)
        return self

    @staticmethod
    @numba.njit
    def categorize(data):
        """Categorize points in 2D grid.

        Count number of sign changes whilst traversing neighbour point loop.

            - 0: minima / maxima point
            - 2: regular point
            - 4: saddle point

        """
        xdim, zdim = data.shape
        o_mask = np.full((xdim, zdim), False)
        x_mask = np.full((xdim, zdim), False)
        for i in numba.prange(1, xdim-1):
            for j in range(1, zdim-1):
                center = data[i, j]
                sign = data[i-1, j+1] >= center
                count = 0
                for k in [(-1, 0), (-1, -1), (0, -1), (1, -1),
                          (1, 0), (1, 1), (0, 1), (-1, 1)]:
                    _sign = data[i+k[0], j+k[1]] >= center
                    if _sign != sign:
                        count += 1
                        sign = _sign
                if count == 0:
                    o_mask[i, j] = True
                if count == 4:
                    x_mask[i, j] = True
        return o_mask, x_mask

    @staticmethod
    @numba.njit
    def minimum(data):
        """Return 2D boolean index indicating locations of data minima."""
        xdim, zdim = data.shape
        mask = np.full((xdim, zdim), False)
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
                mask[i, j] = True
        return mask


if __name__ == '__main__':

    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.plasma.insert(dict(o=(4, 0, 0.5)), delta=0.3)
    coilset.grid.solve(500, 0.05) #[3.2, 8.5, -2.5, 2.5])
    coilset.sloc['Ic'] = -15e6

    grid = coilset.grid

    shape = grid.data.dims['x'], grid.data.dims['z']
    psi, bn = grid.psi.reshape(shape), grid.bn.reshape(shape)
    null = FieldNull().update(psi, bn)

    plt.plot(grid.data.x2d.data[null.index_o],
             grid.data.z2d.data[null.index_o], 'o')
    plt.plot(grid.data.x2d.data[null.index_x],
             grid.data.z2d.data[null.index_x], 'X')

    coilset.plot()
    coilset.grid.plot()
