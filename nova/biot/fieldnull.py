"""Methods for calculating the position and value of x-points and o-points."""
from dataclasses import dataclass, field

import numba
import numpy as np
import xarray

from nova.frame.baseplot import Plot
from nova.geometry.pointloop import PointLoop


@dataclass
class DataNull(Plot):
    """Store sort and remove field nulls."""

    subgrid: bool = True
    data: xarray.Dataset | None = \
        field(repr=False, default_factory=xarray.Dataset)
    loop: np.ndarray | None = field(repr=False, default=None)
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
            return self.update_mask_1d(mask, psi)
        return self.update_mask_2d(mask, psi)

    def update_mask_1d(self, mask, psi):
        """Return masked data dict from 1D input."""
        try:
            index, points = self._index_1d(
                self.data.x.data, self.data.z.data, mask)
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

    def update_mask_2d(self, mask, psi):
        """Return masked data dict from 2D input."""
        try:
            index, points = self._index_2d(
                self.data.x.data, self.data.z.data, mask)
        except IndexError:  # catch empty mask
            index, points = np.empty((0, 2), int), np.empty((0, 2), float)
            return dict(index=index, points=points)
        if self.loop is not None:
            subindex = PointLoop(points).update(self.loop)
            index = index[subindex]
            points = points[subindex]
        if self.subgrid:
            return self.subnull_2d(index, psi)
        data = dict(index=index, points=points)
        data['psi'] = np.array([psi[tuple(i)] for i in index])
        return data

    @staticmethod
    @numba.njit
    def bisect(vector, value):
        """Return the bisect left index, assuming vector is sorted.

        The return index i is such that all e in vector[:i] have e < value,
        and all e in vector[i:] have e >= value.

        Addapted from bisect.bisect_left to allow jit compilation.
        """
        low, high = 0, len(vector)
        while low < high:
            mid = (low + high) // 2
            if vector[mid] < value:
                low = mid + 1
            else:
                high = mid
        return low

    @staticmethod
    @numba.njit
    def quadratic_surface(x_cluster, z_cluster, psi_cluster):
        """Return psi quatratic surface coefficients."""
        coefficient_matrix = np.column_stack(
            (x_cluster**2, z_cluster**2, x_cluster, z_cluster,
             x_cluster*z_cluster, np.ones_like(x_cluster)))
        coefficients = np.linalg.lstsq(coefficient_matrix, psi_cluster)[0]
        return coefficients

    @staticmethod
    @numba.njit
    def null_type(coefficients, atol=1e-12):
        """Return null type.

            - 0: saddle
                :math:`4AB - E^2 < 0`
            - -1: minimum
                :math:`A>0` and :math:`B>0`
            - 1: maximum
                :math:`A<0` and :math:`B<0`

        Raises
        ------
        ValueError
            degenerate surface
        """
        root = 4*coefficients[0]*coefficients[1] - coefficients[4]**2
        if abs(root) < atol:
            raise ValueError('Plane surface')
        if root < 0:
            return 0
        if coefficients[0] > 0 and coefficients[1] > 0:
            return -1
        if coefficients[0] < 0 and coefficients[1] < 0:
            return 1
        raise ValueError('Coefficients form a degenerate surface.')

    @staticmethod
    @numba.njit
    def null_coordinate(coefficients, cluster=None, atol=1e-12):
        """
        Return null coodinates in 2D plane.

        Returns
        -------
        x_coordinate: float
            subgrid field null x_coordinate
        z_coordinate: float
            subgrid field null z_coordinate

        Raises
        ------
        ValueError
            subgrid coordinate outside cluster
        """
        root = 4*coefficients[0]*coefficients[1] - coefficients[4]**2
        x_coordinate = (coefficients[4]*coefficients[3] -
                        2*coefficients[1]*coefficients[2]) / root
        z_coordinate = (coefficients[4]*coefficients[2] -
                        2*coefficients[0]*coefficients[3]) / root
        if cluster is not None:
            for i, coord in enumerate([x_coordinate, z_coordinate]):
                assert coord >= np.min(cluster[i]) - atol
                assert coord <= np.max(cluster[i]) + atol
        return x_coordinate, z_coordinate

    @staticmethod
    @numba.njit
    def null(coef, coords):
        """Return null poloidal flux."""
        return np.array([coords[0]**2, coords[1]**2, coords[0], coords[1],
                        coords[0]*coords[1], 1]) @ coef

    @staticmethod
    def subnull(x_cluster, z_cluster, psi_cluster):
        """Return subgrid null coordinates, value, and type."""
        coef = DataNull.quadratic_surface(x_cluster, z_cluster, psi_cluster)
        coords = DataNull.null_coordinate(coef, (x_cluster, z_cluster))
        psi = DataNull.null(coef, coords)
        null_type = DataNull.null_type(coef)
        return coords, psi, null_type

    def subnull_2d(self, index, psi2d):
        """Return unique field nulls."""
        x2d, z2d = self.data.x2d.values, self.data.z2d.values
        points = np.empty_like(index, dtype=float)
        psi = np.empty(len(index), dtype=float)
        null_type = np.empty(len(index), dtype=int)
        for i, ij in enumerate(index):
            ij = (slice(ij[0]-1, ij[0]+2), slice(ij[1]-1, ij[1]+2))
            x_cluster = x2d[ij].flatten()
            z_cluster = z2d[ij].flatten()
            _psi = psi2d[ij].flatten()
            coef = DataNull.quadratic_surface(x_cluster, z_cluster, _psi)
            null_type[i] = DataNull.null_type(coef)
            points[i] = DataNull.null_coordinate(coef, (x_cluster, z_cluster))
            psi[i] = DataNull.null(coef, points[i])
        points, unique_index = np.unique(points, axis=0, return_index=True)
        return dict(index=index, points=points, psi=psi[unique_index],
                    null_type=null_type[unique_index])

    @staticmethod
    @numba.njit
    def _index_1d(x_coordinate, z_coordinate, mask):
        index = np.where(mask)[0]
        point_number = len(index)
        points = np.empty((point_number, 2), dtype=numba.float64)
        for i in numba.prange(point_number):
            points[i, 0] = x_coordinate[index[i]]
            points[i, 1] = z_coordinate[index[i]]
        return index, points

    @staticmethod
    @numba.njit
    def _index_2d(x_coordinate, z_coordinate, mask):
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
        self.get_axes(axes)
        self.axes.plot(*self.data_o['points'].T, 'C0o')
        self.axes.plot(*self.data_x['points'].T, 'C3X')


@dataclass
class FieldNull(DataNull):
    """Calculate positions of all field nulls."""

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
    @numba.njit
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
    @numba.njit
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
