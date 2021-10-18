"""Geometric methods for FrameSpace class."""
from dataclasses import dataclass, field

import numpy as np

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.polygen import PolyFrame
from nova.electromagnetic.polygeom import PolyGeom


@dataclass
class Geometry(MetaMethod):
    """
    Geometrical methods for FrameSpace.

    Extract geometric features from shapely polygons.
    """

    name = 'geometry'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['section', 'poly'])
    additional: list[str] = field(default_factory=lambda: [
        'dl', 'dt', 'rms', 'area'])
    require_all: bool = field(init=False, repr=False, default=False)
    base: list[str] = field(init=False, default_factory=lambda: [
        'x', 'y', 'z', 'segment', 'dx', 'dy', 'dz'])
    features: list[str] = field(init=False, default_factory=lambda: [
        'x', 'y', 'z', 'dx', 'dy', 'dz', 'area', 'rms'])

    def initialize(self):
        """Update frame polygons and derived geometrical data."""
        rms_unset = (self.frame.rms == 0) & (self.frame.segment == 'ring')
        if sum(rms_unset) > 0:
            index = self.frame.index[rms_unset]
            index_length = len(index)
            section = self.frame.loc[index, 'section'].values
            coords = self.frame.loc[
                index, ['x', 'y', 'z', 'dx', 'dy', 'dz',
                        'segment', 'dl', 'dt']].to_numpy()
            poly = self.frame.loc[index, 'poly'].values
            poly_update = self.frame.loc[index, 'poly'].isna()
            geom = np.empty((index_length, len(self.features)), dtype=float)
            # itterate over index - generate poly as required
            for i in range(index_length):
                polygeom = PolyGeom(poly[i], *coords[i], section[i])
                section[i] = polygeom.section  # inflate section name
                if poly_update[i]:
                    poly[i] = PolyFrame(polygeom.poly, polygeom.section)
                geometry = polygeom.geometry  # extract geometrical features
                geom[i] = [geometry[feature] for feature in self.features]
            if poly_update.any():
                self.frame.loc[index, 'poly'] = poly
            self.frame.loc[index, self.features] = geom
            self.frame.loc[index, 'section'] = section

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.frame.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit
