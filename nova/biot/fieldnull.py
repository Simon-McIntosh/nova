"""Methods for calculating the position and value of x-points and o-points."""
from dataclasses import dataclass, field

import numpy as np
import xarray

from nova import njit
from nova.biot.array import Array
from nova.graphics.plot import Plot
from nova.geometry import select
from nova.geometry.pointloop import PointLoop


@dataclass
class DataNull(Plot, Array):
    """Store sort and remove field nulls."""

    subgrid: bool = True
    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    loop: np.ndarray | None = field(repr=False, default=None)
    array_attrs: list[str] = field(
        default_factory=lambda: ['x', 'z', 'stencil', 'stencil_index'])
    data_o: dict[str, np.ndarray] = field(init=False, default_factory=dict,
                                          repr=False)
    data_x: dict[str, np.ndarray] = field(init=False, default_factory=dict,
                                          repr=False)

    def update_masks(self, mask_o, mask_x, psi):
        """Update null points."""
        for null, mask in zip('ox', [mask_o, mask_x]):
            setattr(self, f'data_{null}', self.update_mask(mask, psi))

    def update_mask(self, mask, psi):
        """Return masked point data dict."""
        if len(psi.shape) == 1:
            return self.update_mask_1d(mask, psi)
        return self.update_mask_2d(mask, psi)

    @staticmethod
    def _empty_mask():
        """Return empty dict structure when all(mask==0)."""
        index, points = np.empty((0, 2), int), np.empty((0, 2), float)
        return dict(index=index, points=points, psi=np.empty(0))

    def _select(self, points, index):
        """Select subset of points within loop when loop is not None."""
        if self.loop is None:
            return points, index
        subindex = PointLoop(points).update(self.loop)
        return points[subindex], index[subindex]

    def update_mask_1d(self, mask, psi):
        """Return masked data dict from 1D input."""
        try:
            index, points = self._index_1d(self['x'], self['z'], mask)
        except IndexError:  # catch empty mask
            return self._empty_mask()
        points, index = self._select(points, index)
        if self.subgrid:
            return {'index': index} | self._subnull_1d(index, psi)
        return {'index': index, 'points': points, 'psi': psi[index]}

    def update_mask_2d(self, mask, psi):
        """Return masked data dict from 2D input."""
        try:
            index, points = self._index_2d(self['x'], self['z'], mask)
        except IndexError:
            return self._empty_mask()
        points, index = self._select(points, index)
        if self.subgrid:
            return {'index': index} | self._subnull_2d(index, psi)
        return {'index': index, 'points': points,
                'psi': np.array([psi[tuple(i)] for i in index])}

    @staticmethod
    def _unique(nulls, decimals=3):
        """Return unique field nulls."""
        points = np.array([null[0] for null in nulls])
        psi = np.array([null[1] for null in nulls])
        null_type = np.array([null[2] for null in nulls])
        points, index = np.unique(points.round(decimals),
                                  axis=0, return_index=True)
        return {'points': points, 'psi': psi[index],
                'null_type': null_type[index]}

    def _subnull_1d(self, index, psi):
        """Return unique field nulls from 1d unstructured grid."""
        stencil_index = self['stencil_index']
        nulls = []
        for i in index:
            stencil_vertex = self['stencil'][select.bisect(stencil_index, i)]
            x_cluster = self['x'][stencil_vertex]
            z_cluster = self['z'][stencil_vertex]
            psi_cluster = psi[stencil_vertex]
            nulls.append(select.subnull(x_cluster, z_cluster, psi_cluster))
        return {'index': index} | self._unique(nulls)

    def _subnull_2d(self, index, psi2d):
        """Return unique field nulls from 2d grid."""
        x_coordinate, z_coordinate = self['x'], self['z']
        nulls = []
        for i, j in index:
            x2d, z2d = np.meshgrid(x_coordinate[i-1:i+2],
                                   z_coordinate[j-1:j+2], indexing='ij')
            x_cluster, z_cluster = x2d.flatten(), z2d.flatten()
            psi_cluster = psi2d[i-1:i+2, j-1:j+2].flatten()
            nulls.append(select.subnull(x_cluster, z_cluster, psi_cluster))
        return dict(index=index) | self._unique(nulls)

    @staticmethod
    @njit(cache=True)
    def _index_1d(x_coordinate, z_coordinate, mask):
        index = np.where(mask)[0]
        point_number = len(index)
        points = np.empty((point_number, 2), dtype=np.float64)
        for i in range(point_number):  # pylint: disable=not-an-iterable
            points[i, 0] = x_coordinate[index[i]]
            points[i, 1] = z_coordinate[index[i]]
        return index, points

    @staticmethod
    @njit(cache=True)
    def _index_2d(x_coordinate, z_coordinate, mask):
        index = np.asarray(list(zip(*np.where(mask))))
        point_number = len(index)
        points = np.empty((point_number, 2), dtype=np.float64)
        for i in range(point_number):  # pylint: disable=not-an-iterable
            points[i, 0] = x_coordinate[index[i][0]]
            points[i, 1] = z_coordinate[index[i][1]]
        return index, points

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
        self.get_axes(axes)
        if self.o_point_number > 0:
            self.axes.plot(*self.data_o['points'].T, 'o',
                           ms=4, mec='C3', mew=1, mfc="none")
        if self.x_point_number > 0:
            self.axes.plot(*self.data_x['points'].T, 'x',
                           ms=6, mec='C3', mew=1, mfc="none")


@dataclass
class FieldNull(DataNull):
    """Calculate positions of all field nulls."""

    @property
    def o_points(self):
        """Return o-point locations."""
        return self.data_o['points']

    @property
    def o_psi(self):
        """Return flux values at o-point locations."""
        return self.data_o['psi']

    @property
    def o_point_number(self):
        """Return o-point number."""
        return len(self.o_psi)

    @property
    def x_points(self):
        """Return x-point locations."""
        return self.data_x['points']

    @property
    def x_psi(self):
        """Return flux values at x-point locations."""
        return self.data_x['psi']

    @property
    def x_point_number(self):
        """Return x-point number."""
        return len(self.x_psi)

    def update_null(self, psi):
        """Update calculation of field nulls."""
        mask_o, mask_x = self.categorize(psi)
        super().update_masks(mask_o, mask_x, psi)

    def categorize(self, psi):
        """Return o-point and x-point masks from loop sign counts."""
        if len(psi.shape) == 1:
            return self.categorize_1d(psi, self.data.stencil.data)
        return self.categorize_2d(psi)

    @staticmethod
    @njit(cache=True)
    def categorize_1d(data, stencil):
        """Categorize points in 1d hexagonal grid.

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
    @njit(cache=True)
    def categorize_2d(data):
        """Categorize points in 2D rectangular grid.

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
        for i in range(1, xdim-1):  # pylint: disable=not-an-iterable
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

    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.firstwall.insert(dict(o=(4, 0, 0.5)), delta=0.3)
    coilset.grid.solve(500, 0.05)
    coilset.sloc['Ic'] = -15e6

    coilset.plot()
    coilset.grid.plot()

    #coilset.grid.plot(levels=np.sort(coilset.grid.x_psi))
