"""Manage single instance polygon data."""
from dataclasses import dataclass, field

import pandas
import numpy as np
import shapely.geometry

from nova.electromagnetic.polygen import polygen, polyshape, boxbound


@dataclass
class PolyGeom:
    """Extract geometrical features from shapely polygons."""

    poly: shapely.geometry.Polygon = field(default=None, repr=False)
    section: str = None
    x_centroid: float = None
    z_centroid: float = None
    length: float = None
    thickness: float = None

    def __post_init__(self):
        """Update section and generate polygon as required."""
        self.update_section()
        self.update_centroid()
        self.generate()

    def update_section(self):
        """Update section name. Extract from poly, inflate str if not found."""
        try:
            self.section = self.poly.name
        except AttributeError:
            if self.section is not None:
                self.section = polyshape[self.section]  # inflate shorthand

    def update_centroid(self):
        """Nulify centroid if section not defined in polyshape."""
        if self.section not in polyshape:
            self.x_centroid = self.z_centroid = None

    def generate(self):
        """Generate polygon if not set."""
        if pandas.isna(self.poly):
            self.poly = polygen(self.section)(self.x_centroid, self.z_centroid,
                                              self.length, self.thickness)

    @property
    def geometry(self) -> dict[str, float]:
        """Return geometrical features."""
        centroid = self.centroid
        bbox = self.bbox
        return {'x': centroid[0], 'z': centroid[1], 'dl': self.length,
                'dt': self.thickness, 'dx': bbox[0], 'dz': bbox[1],
                'area': self.area, 'rms': self.rms,
                'poly': self.poly, 'section': self.poly.name}

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
            self.x_centroid = self.poly.centroid.x  # update x centroid
        if self.z_centroid is None:
            self.z_centroid = self.poly.centroid.y  # update z centroid
        return self.x_centroid, self.z_centroid

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
        cross_section : str
            Cross section descriptor.
        x_center : float
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
        if self.section == 'circle':
            return np.sqrt(self.x_centroid**2 + self.length**2 / 16)  # circle
        if self.section in ['square', 'rectangle']:
            return np.sqrt(self.x_centroid**2 + self.length**2 / 12)  # square
        if self.section == 'skin':
            return np.sqrt((self.length**2 * self.thickness**2 / 24
                            - self.length**2 * self.thickness / 8
                            + self.length**2 / 8 + self.x_centroid**2))
        return (shapely.ops.transform(
            lambda x, z: (x**2, z), self.poly).centroid.x)**0.5
