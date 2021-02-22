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
    section: bool = False

    _required_attributes = ['x', 'z', 'rms', 'dl', 'dt', 'dx', 'dz', 'dA',
                            'section', 'poly', 'patch']

    def __post_init__(self):
        """Update additional attributes."""
        self.update_section()

    def update_section(self):
        """Update section flag."""
        self.section = 'section' in self.frame.columns
        if self.section:
            self.frame.metadata = {'additional': self._required_attributes}

    def generate(self):
        """Generate polygons based on coil geometroy and cross section."""
        if self.section:
            for index in self.frame.index[self.frame.poly.isna()]:
                section = self.frame.loc[index, 'section']
                poly = polygen(section)(
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
            section = self.frame.at[key, 'section']
            length, thickness = self.frame.dl[i], self.frame.dt[i]
            area = poly.area  # update polygon area
            if area == 0:
                raise ValueError(
                    f'zero area polygon entered for coil {index}\n'
                    f'cross section: {section}\n'
                    f'length {length}\nthickness {thickness}')
            x_center = poly.centroid.x  # update x centroid
            z_center = poly.centroid.y  # update z centroid
            self.frame.loc[key, ['x', 'z', 'dA']] = x_center, z_center, area
            bounds = poly.bounds
            self.frame.at[key, 'dx'] = bounds[2]-bounds[0]
            self.frame.at[key, 'dz'] = bounds[3]-bounds[1]
            self.frame.at[key, 'rms'] = root_mean_square(
                section, x_center, length, thickness, poly)
        #if len(index) != 0:
        #    self.update_dataframe = ['x', 'z', 'dx', 'dz', 'rms']
