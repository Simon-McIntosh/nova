"""Manage single instance polygon data."""
from dataclasses import dataclass, field
from typing import Union

import pandas
import numpy as np
import shapely.geometry

from nova.electromagnetic.polygen import polygen, polyshape, boxbound


@dataclass
class Poly:
    """Order Poly variable."""

    poly: shapely.geometry.Polygon = field(default=None, repr=False)


@dataclass
class Geom(Poly):
    """Manage segment geometrical parameters."""

    x_coordinate: float = None
    y_coordinate: float = None
    z_coordinate: float = None
    delta_x: float = None
    delta_y: float = None
    delta_z: float = None
    segment: str = 'circle'

    @property
    def centroid(self):
        """
        Return segment centroid.

        Returns
        -------
        centroid : list[float, float, float]
            Segment centroid [x, y, z] coordinates.

        """
        if self.segment == 'circle':  # centered toroildal circle
            if self.x_coordinate is None:
                self.x_coordinate = self.poly.centroid.x  # update x centroid
            if self.y_coordinate is None:
                self.y_coordinate = 0
            if self.z_coordinate is None:
                self.z_coordinate = self.poly.centroid.y  # update z centroid
        return self.x_coordinate, self.y_coordinate, self.z_coordinate


@dataclass
class PolyGeom(Geom):
    """Extract section geometrical features from shapely polygons."""

    length: float = None
    thickness: float = None
    section: str = 'rectangle'

    def __post_init__(self):
        """Generate polygon as required."""
        self.update_section()
        if self.segment == 'circle':
            self.generate()

    def generate(self):
        """Generate poloidal polygon for circular filaments."""
        if self.section not in polyshape:  # clear poloidal coordinate
            self.x_coordinate = self.z_coordinate = None
        if pandas.isna(self.poly):
            self.poly = polygen(self.section)(
                *self.centroid[::2], self.length, self.thickness)
        self.delta_x = self.bbox[0]
        self.delta_y = 2*np.pi*self.centroid[0]  # diameter
        self.delta_z = self.bbox[1]

    def update_section(self):
        """Update section name. Extract from poly, inflate str if not found."""
        try:
            self.section = self.poly.name
        except AttributeError:
            if self.section is not None:
                self.section = polyshape[self.section]  # inflate shorthand

    @property
    def area(self):
        """Return polygon area."""
        if self.section == 'circle':
            return np.pi * self.length**2 / 4  # circle
        if self.section == 'square':
            return self.length**2  # square
        if self.section == 'rectangle':
            return self.length * self.thickness  # rectangle
        if self.section == 'skin':  # thickness = 1-r/R
            return np.pi*self.length**2*self.thickness / 4 * \
                (2 - self.thickness**2)
        return self.poly.area

    @property
    def bbox(self):
        """Return width and height of polygon bounding box."""
        if self.section in polyshape and self.section != 'skin' and \
                self.length is not None and \
                self.thickness is not None:
            if self.section in ['circle', 'square']:
                self.length = self.thickness = boxbound(self.length,
                                                        self.thickness)
            return self.length, self.thickness
        bounds = self.poly.bounds
        width = bounds[2]-bounds[0]
        height = bounds[3]-bounds[1]
        return width, height

    @property
    def rms(self):
        """
        Return section root mean square radius.

        Perform fast rms calculation for defined sections.
        Numerical calculation exicuted on section is not a base geometory.

        Parameters
        ----------
        section : str
            Cross section descriptor.
        centroid_radius : float
            Radial coordinate of geometric centroid.
        length : float
            First characteristic dimension, dl.
        thickness : float
            Second characteristic dimension, dt.
        poly : shapely.polygon, optional
            Polygon for numerical calculation if not in
            [circle, square, rectangle, skin]. The default is None.

        Returns
        -------
        radius : float
            Root mean square radius (uniform current density current center).

        """
        if self.segment != 'circle':
            return -1
        centroid_radius = self.centroid[0]
        if self.section == 'circle':
            return np.sqrt(centroid_radius**2 + self.length**2 / 16)  # circle
        if self.section in ['square', 'rectangle']:
            return np.sqrt(centroid_radius**2 + self.length**2 / 12)  # square
        if self.section == 'skin':
            return np.sqrt((self.length**2 * self.thickness**2 / 24
                            - self.length**2 * self.thickness / 8
                            + self.length**2 / 8 + centroid_radius**2))
        return (shapely.ops.transform(
            lambda x, z: (x**2, z), self.poly).centroid.x)**0.5

    @property
    def geometry(self) -> dict[str, float]:
        """Return geometrical features."""
        centroid = self.centroid
        return {'x': centroid[0], 'y': centroid[1], 'z': centroid[2],
                'dl': self.length, 'dt': self.thickness,
                'dx': self.delta_x, 'dy': self.delta_y, 'dz': self.delta_z,
                'area': self.area, 'rms': self.rms,
                'poly': self.poly, 'section': self.section}
