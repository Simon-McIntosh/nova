"""Geometric methods for Frame and FrameArray."""
from dataclasses import dataclass, field

import pandas
import numpy as np
import shapely.geometry

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.polygen import (
    polygen, polyframe, polyshape, boxbound
    )
from nova.electromagnetic.dataframe import DataFrame


@dataclass
class PolyGeom:
    """Extract geometrical features from shapely polygons."""

    section: str
    x_centroid: float = field(default=None)
    z_centroid: float = field(default=None)
    length: float = field(default=None)
    thickness: float = field(default=None)
    scale: float = 1
    polygon: shapely.geometry.Polygon = field(default=None, repr=False)

    def __post_init__(self):
        """Update section and generate polygon as required."""
        self.section = polyshape[self.section]  # inflate shorthand
        self.length *= self.scale
        self.thickness *= self.scale
        if pandas.isna(self.polygon):  # generate polygon if not set
            self.polygon = polygen(self.section)(
                self.x_centroid, self.z_centroid,
                self.length, self.thickness)

    def extract(self):
        """Return geometrical feature set."""
        x_centroid, z_centroid = self.centroid
        width, height = self.bounding_box
        area = self.area
        rms = self.root_mean_square
        if area == 0:
            raise ValueError(f'zero area polygon \n'
                             f'cross section: {self.section}\n'
                             f'length {self.length}\n'
                             f'thickness {self.thickness}')
        return x_centroid, z_centroid, width, height, area, rms

    @property
    def centroid(self):
        """
        Return polygon centroid.

        Returns
        -------
        x_centroid : float
            Polygon centroid x-coordinate.
        z_centroid : float
            Polygon centroid z-coordinate.
        """
        if self.x_centroid is None:
            self.x_centroid = self.polygon.centroid.x  # update x centroid
        if self.x_centroid is None:
            self.z_centroid = self.polygon.centroid.y  # update z centroid
        return self.x_centroid, self.z_centroid

    @property
    def area(self):
        """Return polygon area."""
        if self.section == 'circle':
            return np.pi * self.length**2 / 4  # circle
        if self.section in ['square', 'rectangle']:
            return self.length * self.thickness  # square
        if self.section == 'skin':  # thickness = 1-r/R
            return 4*np.pi*self.thickness / self.length**2 * \
                (2 - self.thickness**2)
        return self.polygon.area

    @property
    def bounding_box(self):
        """Return width and height of polygon bounding box."""
        if self.section in polyshape:
            if self.section in ['circle', 'square']:
                self.length = self.thickness = boxbound(self.length,
                                                        self.thickness)
            return self.length, self.thickness
        bounds = self.polygon.bounds
        width = bounds[2]-bounds[0]
        height = bounds[3]-bounds[1]
        return width, height

    @property
    def root_mean_square(self):
        """
        Return section root mean square radius.

        Perform fast rms calculation for defined sections.
        Numerical calculation exicuted on section is not a base geometory.

        Parameters
        ----------
        cross_section : str
            Cross section descriptor.
        x_center : float
            Radial coordinate of geometric centroid.
        length : float
            First characteristic dimension, dl.
        thickness : float
            Second characteristic dimension, dt.
        polygon : shapely.polygon, optional
            Polygon for numerical calculation if not in
            [circle, square, rectangle, skin]. The default is None.

        Returns
        -------
        radius : float
            Root mean square radius (uniform current density current center).

        """
        if self.section == 'circle':
            return np.sqrt(self.x_centroid**2 + self.length**2 / 16)  # circle
        if self.section in ['square', 'rectangle']:
            return np.sqrt(self.x_centroid**2 + self.length**2 / 12)  # square
        if self.section == 'skin':
            return np.sqrt((self.length**2 * self.thickness**2 / 24
                            - self.length**2 * self.thickness / 8
                            + self.length**2 / 8 + self.x_centroid**2))
        return (shapely.ops.transform(
            lambda x, z: (x**2, z), self.polygon).centroid.x)**0.5


@dataclass
class Polygon:
    """Extract geometric features from shapely polygons."""

    frame: DataFrame = field(repr=False)
    features: list[str] = field(init=False, default_factory=lambda: [
        'x', 'z', 'dx', 'dz', 'dA', 'rms'])

    def update(self):
        """Update frame polygons and derived data."""
        rms_unset = self.frame.rms == 0
        if sum(rms_unset) > 0:
            index = self.frame.index[rms_unset]
            index_length = len(index)
            section = self.frame.loc[index, 'section'].values
            coords = self.frame.loc[
                index, ['x', 'z', 'dl', 'dt', 'scale']].to_numpy()
            polygon = self.frame.loc[index, 'poly'].values
            polygon_update = self.frame.loc[index, 'poly'].isna()
            geom = np.empty((index_length, len(self.features)), dtype=float)
            # itterate over index - generate polygon as required
            for i in range(index_length):
                polygeom = PolyGeom(section[i], *coords[i], polygon[i])
                section[i] = polygeom.section  # inflate section name
                if polygon_update[i]:
                    polygon[i] = polyframe(polygeom.polygon)
                geom[i] = polygeom.extract()
            if polygon_update.any():
                self.frame.loc[index, 'poly'] = polygon
            self.frame.loc[index, self.features] = geom
            self.frame.loc[index, 'section'] = section


@dataclass
class Geometry(MetaMethod):
    """Geometrical methods for Frame."""

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: [
        'x', 'z', 'section'])
    additional: list[str] = field(default_factory=lambda: [
        'dl', 'dt', 'scale', 'rms', 'dx', 'dz', 'dA', 'poly'])
    require_all: bool = field(repr=False, default=True)
    polygon: Polygon = field(init=False, repr=False)

    def __post_init__(self):
        """Launch polygon instance."""
        if self.generate:
            self.polygon = Polygon(self.frame)
        super().__post_init__()

    def initialize(self):
        """Update frame polygons and derived geometrical data."""
        self.polygon.update()

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.frame.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit
