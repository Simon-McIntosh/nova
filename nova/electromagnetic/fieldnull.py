"""Methods for calculating the position and value of x-points and o-points."""
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt

from nova.electromagnetic.baseplot import Axes
from nova.geometry.pointloop import PointLoop


@dataclass
class DataNull(Axes):
    """Store sort and remove field nulls."""

    x_coordinate: npt.ArrayLike = field(repr=False)
    z_coordinate: npt.ArrayLike = field(repr=False)
    loop: npt.ArrayLike = field(repr=False, default=None)
    data_o: dict[str, np.ndarray] = field(init=False, default_factory=dict,
                                          repr=False)
    data_x: dict[str, np.ndarray] = field(init=False, default_factory=dict,
                                          repr=False)

    @property
    def o_point(self):
        """Return o-point locations."""
        return self.data_o['points']

    @property
    def o_psi(self):
        """Return flux values at o-point locations."""
        return self.data_o['psi']

    @property
    def o_point_number(self):
        """Return o-point number."""
        return len(self.data_o['points'])

    @property
    def x_point(self):
        """Return x-point locations."""
        return self.data_x['points']

    @property
    def x_psi(self):
        """Return flux values at x-point locations."""
        return self.data_x['psi']

    @property
    def x_point_number(self):
        """Return x-point number."""
        return len(self.data_x['points'])

    def update_masks(self, mask_o, mask_x, psi):
        """Update null points."""
        for null, mask in zip('ox', [mask_o, mask_x]):
            setattr(self, f'data_{null}', self.update_mask(mask, psi))

    def update_mask(self, mask, psi):
        """Return masked point data dict."""
        if len(psi.shape) == 1:
            return self.update_mask_1D(mask, psi)
        return self.update_mask_2D(mask, psi)

    def update_mask_1D(self, mask, psi):
        """Return masked data dict from 1D input."""
        try:
            index, points = self._index_1D(
                self.x_coordinate, self.z_coordinate, mask)
        except IndexError:  # catch empty mask
            index, points = np.empty((0, 2), int), np.empty((0, 2), float)
            return dict(index=index, points=points)
        if self.loop is not None:
            subindex = PointLoop(points).update(self.loop)
            index = index[subindex]
            points = points[subindex]
        data = dict(index=index, points=points)
        data['psi'] = psi[index]
        return data

    def update_mask_2D(self, mask, psi):
        """Return masked data dict from 2D input."""
        try:
            index, points = self._index_2D(
                self.x_coordinate, self.z_coordinate, mask)
        except IndexError:  # catch empty mask
            index, points = np.empty((0, 2), int), np.empty((0, 2), float)
            return dict(index=index, points=points)
        if self.loop is not None:
            subindex = PointLoop(points).update(self.loop)
            index = index[subindex]
            points = points[subindex]
        data = dict(index=index, points=points)
        data['psi'] = np.array([psi[tuple(i)] for i in index])
        return data

    @staticmethod
    @numba.njit
    def _index_1D(x_coordinate, z_coordinate, mask):
        index = np.where(mask)[0]
        point_number = len(index)
        points = np.empty((point_number, 2), dtype=numba.float64)
        for i in numba.prange(point_number):
            points[i, 0] = x_coordinate[index[i]]
            points[i, 1] = z_coordinate[index[i]]
        return index, points

    @staticmethod
    @numba.njit
    def _index_2D(x_coordinate, z_coordinate, mask):
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

    x_coordinate: npt.ArrayLike = field(repr=False, default=None)
    z_coordinate: npt.ArrayLike = field(repr=False, default=None)
    loop: npt.ArrayLike = None
    stencil: npt.ArrayLike = field(repr=False, default=None)

    def update_null(self, psi):
        """Update calculation of field nulls."""
        mask_o, mask_x = self.categorize(psi)
        super().update_masks(mask_o, mask_x, psi)

    def categorize(self, psi):
        """Return o-point and x-point masks from loop sign counts."""
        if len(psi.shape) == 1:
            return self.categorize_1D(psi, self.stencil)
        return self.categorize_2D(psi)

    @staticmethod
    @numba.njit
    def categorize_1D(data, stencil):
        """Categorize points in 1D hexagonal grid.

        Count number of sign changes whilst traversing neighbour point loop.

            - 0: minima / maxima point
            - 2: regular point
            - 4: saddle point

        From On detecting all saddle points in 2D images, A. Kuijper

        """
        npoint = len(data)
        o_mask = np.full(npoint, False)
        x_mask = np.full(npoint, False)
        for index in stencil:
            center = data[index[0]]
            sign = data[index[-1]] > center
            count = 0
            for k in range(1, 7):
                _sign = data[index[k]] > center
                if _sign != sign:
                    count += 1
                    sign = _sign
            if count == 0:
                o_mask[index[0]] = True
            if count == 4:
                x_mask[index[0]] = True
        return o_mask, x_mask

    @staticmethod
    @numba.njit
    def categorize_2D(data):
        """Categorize points in 2D grid.

        Count number of sign changes whilst traversing neighbour point loop.

            - 0: minima / maxima point
            - 2: regular point
            - 4: saddle point

        From On detecting all saddle points in 2D images, A. Kuijper

        """
        xdim, zdim = data.shape
        o_mask = np.full((xdim, zdim), False)
        x_mask = np.full((xdim, zdim), False)
        stencil = [(-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)]
        #  stencil = [(-1, 0), (-1, -1), (0, -1), (1, 0), (1, 1), (0, 1)]
        for i in numba.prange(1, xdim-1):
            for j in range(1, zdim-1):
                center = data[i, j]
                sign = data[i+stencil[-1][0], j+stencil[-1][1]] > center
                count = 0
                #  use 6-point stencil
                for k in stencil:
                    _sign = data[i+k[0], j+k[1]] > center
                    if _sign != sign:
                        count += 1
                        sign = _sign
                if count == 0:
                    o_mask[i, j] = True
                if count == 4:
                    x_mask[i, j] = True
        return o_mask, x_mask


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.plasma.insert(dict(o=(4, 0, 0.5)), delta=0.3)
    coilset.grid.solve(500, 0.05)
    coilset.sloc['Ic'] = -15e6

    coilset.plot()
    coilset.grid.plot()
