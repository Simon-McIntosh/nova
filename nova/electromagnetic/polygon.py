"""Geometric methods for Frame and FrameArray."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.polygen import polygen, polyframe, root_mean_square

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame

# pylint:disable=unsubscriptable-object


@dataclass
class Polygon(MetaMethod):
    """Geometrical methods for Frame."""

    frame: Frame = field(repr=False)
    required: list[str] = field(default_factory=lambda: [
        'x', 'z', 'section'])
    additional: list[str] = field(default_factory=lambda: [
        'dl', 'dt', 'rms', 'dx', 'dz', 'dA', 'poly', 'patch'])
    require_all: bool = True

    def initialize(self):
        """Init polygons based on coil geometroy and cross section."""
        for index in self.frame.index[self.frame.poly.isna()]:
            section = self.frame.loc[index, 'section']
            poly = polygen(section)(
                *self.frame.loc[index, ['x', 'z', 'dl', 'dt']])
            self.frame.loc[index, 'poly'] = polyframe(poly)
        self.build()

    def build(self, index=None):
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
        data = {attr: getattr(self.frame, attr).copy()
                for attr in ['x', 'z', 'dx', 'dz', 'dA', 'rms']}
        for key in index:
            i = self.frame.index.get_loc(key)
            poly = self.frame.poly[i]
            section = self.frame.section[i]
            length, thickness = self.frame.dl[i], self.frame.dt[i]
            data['x'][i] = poly.centroid.x  # update x centroid
            data['z'][i] = poly.centroid.y  # update z centroid
            bounds = poly.bounds
            data['dx'][i] = bounds[2]-bounds[0]
            data['dz'][i] = bounds[3]-bounds[1]
            data['dA'][i] = poly.area  # update polygon area
            if data['dA'][i] == 0:
                raise ValueError(
                    f'zero area polygon entered for coil {index}\n'
                    f'cross section: {section}\n'
                    f'length {length}\nthickness {thickness}')
            data['rms'][i] = root_mean_square(
                section, data['x'][i], length, thickness, poly)
        for attr in data:  # update frame
            setattr(self.frame, attr, data[attr])

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.frame.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit
