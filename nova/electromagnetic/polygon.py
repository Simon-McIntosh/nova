"""Geometric methods for Frame and FrameArray."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

import pandas

from nova.electromagnetic.polygen import polygen, polyframe, root_mean_square

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame, FrameArray

# pylint:disable=unsubscriptable-object


@dataclass
class Polygon:
    """Geometrical methods for Frame and FrameArray."""

    frame: Union[Frame, FrameArray]

    def __post_init__(self):
        """Update additional attributes."""
        print(self.frame.metaframe.required)
        self.frame.metadata = {
            'additional': ['x', 'z', 'rms', 'dl', 'dt', 'dx', 'dz', 'dA',
                           'cross_section', 'poly', 'patch']}
        print(self.frame.metaframe.additional)

    def generate(self):
        """Generate polygons based on coil geometroy and cross section."""
        if 'poly' in self.frame.columns:
            for index in self.frame.index[self.frame.poly.isna()]:
                cross_section = self.frame.loc[index, 'cross_section']
                poly = polygen(cross_section)(
                    *self.frame.loc[index, ['x', 'z', 'dl', 'dt']])
                self.frame.loc[index, 'poly'] = polyframe(poly)
            self.update()

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.frame.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit

    def update(self, index=None):
        """
        Update polygon derived attributes.

        Derived attributes:
            - x, z, dx, dz : float
                coil centroid and bounding box

        Parameters
        ----------
        index : str or array-like or Index, optional
            Frame subindex. The default is None (all coils).

        Raises
        ------
        ValueError
            Zero cross-sectional area.

        Returns
        -------
        None.

        """
        if index is None:
            index = self.frame.index[(self.frame.rms == 0) &
                                     (~self.frame.poly.isna())]
        elif not pandas.api.types.is_list_like(index):
            index = [index]
        for key in index:
            i = self.frame.index.get_loc(key)
            poly = self.frame.at[key, 'poly']
            cross_section = self.frame.at[key, 'cross_section']
            length, thickness = self.frame.dl[i], self.frame.dt[i]
            area = poly.area  # update polygon area
            if area == 0:
                raise ValueError(
                    f'zero area polygon entered for coil {index}\n'
                    f'cross section: {cross_section}\n'
                    f'length {length}\nthickness {thickness}')
            x_center = poly.centroid.x  # update x centroid
            z_center = poly.centroid.y  # update z centroid
            self.frame.x[i] = x_center
            self.frame.z[i] = z_center
            self.frame.loc[key, 'dA'] = area
            bounds = poly.bounds
            self.frame.dx[i] = bounds[2] - bounds[0]
            self.frame.dz[i] = bounds[3] - bounds[1]
            self.frame.rms[i] = root_mean_square(cross_section, x_center,
                                                 length, thickness, poly)
        #if len(index) != 0:
        #    self.update_dataframe = ['x', 'z', 'dx', 'dz', 'rms']
