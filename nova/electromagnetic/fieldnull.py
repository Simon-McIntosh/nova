
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt

from nova.electromagnetic.baseplot import Axes
from nova.geometry.pointloop import PointLoop
from nova.geometry.polygon import Polygon


@dataclass
class DataNull(Axes):
    """Store sort and remove field nulls."""

    coords: npt.ArrayLike = field(repr=False)
    loop: npt.ArrayLike = field(repr=False, default=None)
    data_o: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    data_x: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def check_null(self):
        return

    def __getattribute__(self, attr):
        if attr == 'data_x':
            self.check_null()
            print(attr, 'trigger update here')
        return super().__getattribute__(attr)

    @property
    def o_point(self):
        """Return o-point locations."""
        return self.data_o['points']

    @property
    def o_point_number(self):
        """Return o-point number."""
        return len(self.data_o['points'])

    @property
    def x_point(self):
        """Return x-point locations."""
        return self.data_x['points']

    @property
    def x_point_number(self):
        """Return x-point number."""
        return len(self.data_x['points'])

    def update_masks(self, mask_o, mask_x, **field_data):
        """Update null points."""
        for null, mask in zip('ox', [mask_o, mask_x]):
            setattr(self, f'data_{null}', self.update_mask(mask, **field_data))

    def update_mask(self, mask, **field_data):
        """Return masked data dict."""
        index, points = self._index(*self.coords, mask)
        if self.loop is not None:
            subindex = PointLoop(points).update(self.loop)
            index = index[subindex]
            points = points[subindex]
        data = dict(index=index, points=points)
        for attr in field_data:
            if (value := field_data[attr]) is not None:
                data[attr] = np.array([value[tuple(i)] for i in index])
        return data

    @staticmethod
    @numba.njit
    def _index(x_coordinate, z_coordinate, mask):
        index = np.asarray([(i, j) for i, j in zip(*np.where(mask))])
        point_number = len(index)
        points = np.empty((point_number, 2), dtype=numba.float64)
        for i in numba.prange(point_number):
            points[i, 0] = x_coordinate[index[i][0]]
            points[i, 1] = z_coordinate[index[i][1]]
        return index, points

    def sort(self):
        """Sort data."""
        raise NotImplementedError

    def delete(self, null: str, index):
        """Delete elements in data specified by index.

        Parameters
        ----------
            index: slice, int or array of ints
                index to remove.

        """
        data = getattr(self, f'data_{null}')
        for attr in data:
            data[attr] = np.delete(data[attr], index, axis=0)

    def plot(self, axes=None):
        """Plot null points."""
        self.axes = axes
        self.axes.plot(*self.data_o['points'].T, 'C0o')
        self.axes.plot(*self.data_x['points'].T, 'C3X')


@dataclass
class FieldNull(DataNull):
    """Calculate positions of all field nulls."""

    coords: npt.ArrayLike
    loop: npt.ArrayLike = None

    def update_null(self, flux, bnorm=None):
        """Update calculation of field nulls."""
        mask_o, mask_x = self.categorize(flux)
        if field is not None:
            mask_x &= self.minimum(bnorm)
        super().update_masks(mask_o, mask_x, flux=flux)

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

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.plasma.insert(dict(o=(4, 0, 0.5)), delta=0.3)
    coilset.grid.solve(500, 0.05) #[3.2, 8.5, -2.5, 2.5])
    coilset.sloc['Ic'] = -15e6

    grid = coilset.grid

    shape = grid.data.dims['x'], grid.data.dims['z']
    psi, bn = grid.psi.reshape(shape), grid.bn.reshape(shape)

    loop = Polygon(dict(o=[4, 0, 30.5])).boundary

    null = FieldNull(grid.coords, loop=None)

    null.update_null(psi, bn)
    null.plot()

    coilset.plot()
    coilset.grid.plot()
