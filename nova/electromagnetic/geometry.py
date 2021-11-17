"""Geometric methods for FrameSpace class."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas
import vedo

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.polygeom import PolyGeom
from nova.electromagnetic.polygen import PolyFrame
from nova.electromagnetic.volume import TriShell, Ring
from nova.electromagnetic.vtkgen import VtkFrame


@dataclass
class VtkGeo(MetaMethod):
    """Volume vtk geometry."""

    name = 'vtkgeo'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['vtk'])
    additional: list[str] = field(
        default_factory=lambda: [*TriShell.features, 'body',
                                 'segment', 'section', 'poly'])
    features: list[str] = field(
        init=False, default_factory=lambda: TriShell.features)
    qhull: ClassVar[list[str]] = ['panel']
    geom: ClassVar[list[str]] = ['panel', 'stl', 'insert', '']

    def initialize(self):
        """Init vtk data."""
        index = self.frame.index[~self.frame.geotype('Geo', 'vtk') &
                                 ~self.frame.geotype('Json', 'vtk') &
                                 ~pandas.isna(self.frame.vtk)]
        if (index_length := len(index)) > 0:
            frame = self.frame.loc[index, :].copy()
            for i in range(index_length):
                tri = TriShell(frame.vtk[i], qhull=frame.body[i] in self.qhull)
                mesh = vedo.Mesh([tri.vtk.points(), tri.vtk.cells()],
                                 c=tri.vtk.c(), alpha=tri.vtk.opacity())
                frame.loc[frame.index[i], 'vtk'] = VtkFrame(mesh)
                if frame.body[i] in self.geom:
                    frame.loc[frame.index[i], self.features] = tri.geom
                    frame.loc[frame.index[i], ['segment', 'section']] = ''
                    frame.loc[frame.index[i], 'poly'] = tri.poly
                else:
                    frame.loc[frame.index[i], 'volume'] = tri.volume
            self.frame.loc[index, :] = frame
        self.generate_vtk()

    def generate_vtk(self):
        """Generate vtk data from poly."""
        index = self.frame.index[~self.frame.geotype('Geo', 'vtk') &
                                 self.frame.geotype('Geo', 'poly')]
        if len(index) > 0:
            self.frame.loc[index, 'vtk'] = \
                [Ring(poly) for poly in self.frame.loc[index, 'poly'].values]


@dataclass
class PolyGeo(MetaMethod):
    """
    Polygon geometrical methods for FrameSpace.

    Extract geometric features from shapely polygons.
    """

    name = 'polygeo'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: [
        'segment', 'section', 'poly'])
    additional: list[str] = field(default_factory=lambda: [
        'dl', 'dt', 'rms', 'area'])
    require_all: bool = field(init=False, repr=False, default=False)
    base: list[str] = field(init=False, default_factory=lambda: [
        'x', 'y', 'z', 'segment', 'dx', 'dy', 'dz'])
    features: list[str] = field(init=False, default_factory=lambda: [
        'x', 'y', 'z', 'dx', 'dy', 'dz', 'area', 'rms'])

    def initialize(self):
        """Init sectional polygon data."""
        index = self.frame.index[~self.frame.geotype('Geo', 'poly') &
                                 ~self.frame.geotype('Json', 'poly') &
                                 (self.frame.segment != '') &
                                 (self.frame.section != '')]
        if (index_length := len(index)) > 0:
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
