"""Manage single instance polygon data."""
from dataclasses import dataclass, field
from typing import Union

import pandas
import numpy as np
import shapely.geometry

from nova.electromagnetic.polygen import PolyGen, PolyFrame
from nova.utilities.pyplot import plt


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
    segment: str = 'ring'

    @property
    def centroid(self):
        """
        Return segment centroid.

        Returns
        -------
        centroid : list[float, float, float]
            Segment centroid [x, y, z] coordinates.

        """
        if self.segment == 'ring':  # centered toroildal disk
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
        if self.segment == 'ring':
            self.generate()

    def generate(self):
        """Generate poloidal polygon for circular filaments."""
        if self.section not in PolyGen.polyshape:  # clear
            self.x_coordinate = self.z_coordinate = None
        if pandas.isna(self.poly):
            self.poly = PolyGen(self.section)(
                *self.centroid[::2], self.length, self.thickness)
        self.delta_x = self.bbox[0]
        self.delta_y = 2*np.pi*self.centroid[0]  # diameter
        self.delta_z = self.bbox[1]

    def update_section(self):
        """Update section name. Extract from poly, inflate str if not found."""
        try:
            self.section = self.poly.name
        except AttributeError:
            if self.section is not None:  # inflate shorthand
                self.section = PolyGen.polyshape[self.section]

    @property
    def area(self):
        """Return polygon area."""
        if self.section == 'disk':
            return np.pi * self.length**2 / 4  # disk
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
        if self.section in PolyGen.polyshape and self.section != 'skin' and \
                self.length is not None and \
                self.thickness is not None:
            if self.section in ['disk', 'square']:
                self.length = self.thickness = PolyGen.boxbound(
                    self.length, self.thickness)
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
            [disk, square, rectangle, skin]. The default is None.

        Returns
        -------
        radius : float
            Root mean square radius (uniform current density current center).

        """
        if self.segment != 'ring':
            return -1
        centroid_radius = self.centroid[0]
        if self.section == 'disk':
            return np.sqrt(centroid_radius**2 + self.length**2 / 16)  # disk
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


@dataclass
class Polygon(PolyGeom):
    """Generate bounding polygon."""

    poly: Union[dict[str, list[float]], list[float],
                shapely.geometry.Polygon] = field(repr=False)

    def __post_init__(self):
        """Generate bounding polygon."""
        self.poly = self.generate_poly(self.poly)
        super().__post_init__()  # generate geometric parameters

    def generate_poly(self, poly):
        """
        Generate polygon.

        Parameters
        ----------
        poly :
            - shapely.geometry.Polygon
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        Raises
        ------
        IndexError
            Malformed bounding box, shape is not (4,).
            Malformed bounding loop, shape is not (n, 2).

        Returns
        -------
        polygon : Polygon
            Limit boundary.

        """
        if isinstance(poly, shapely.geometry.Polygon):
            return PolyFrame(poly, name='polygon')
        if isinstance(poly, dict):
            polys = [PolyGen(section)(*poly[section]) for section in poly]
            if len(polys) == 1:
                self.section = polys[0].name
                values = poly[next(iter(poly))]
                for attr, value in zip(['x_centroid', 'z_centroid',
                                        'length', 'thickness'], values):
                    setattr(self, attr, value)
                return polys[0]
            poly = PolyFrame(shapely.ops.unary_union(polys), name='polygon')
            if not poly.is_valid:
                raise AttributeError('non-overlapping polygons specified in '
                                     f'{self.poly}')
            return poly
        loop = np.array(poly)  # to numpy array
        if loop.ndim == 1:   # poly bounding box
            if len(loop) == 4:  # [xmin, xmax, zmin, zmax]
                xlim, zlim = loop[:2], loop[2:]
                self.x_centroid, self.z_centroid = np.mean(xlim), np.mean(zlim)
                self.length = np.diff(xlim)[0]
                self.thickness = np.diff(zlim)[0]
                if np.isclose(self.length, self.thickness):
                    self.section = 'square'
                else:
                    self.section = 'rectangle'
                return PolyGen(self.section)(self.x_centroid, self.z_centroid,
                                             self.length, self.thickness)
            raise IndexError('malformed bounding box\n'
                             f'loop: {loop}\n'
                             'require [xmin, xmax, zmin, zmax]')
        if loop.shape[1] != 2:
            loop = loop.T
        if loop.ndim == 2 and loop.shape[1] == 2:  # loop
            return PolyFrame(shapely.geometry.Polygon(loop), name='loop')
        raise IndexError('malformed bounding loop\n'
                         f'shape(loop): {loop.shape}\n'
                         'require (n,2)')

    def orient(self):
        """Return coerced polygon boundary as a positively oriented curve."""
        self.poly = shapely.geometry.polygon.orient(self.poly)

    def plot_exterior(self):
        """Plot boundary polygon."""
        plt.plot(*self.poly.exterior.xy)

    @property
    def xlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.poly.bounds[::2]

    @property
    def width(self) -> float:
        """Return polygon bounding box width."""
        return np.diff(self.xlim)[0]

    @property
    def zlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.poly.bounds[1::2]

    @property
    def height(self) -> float:
        """Return polygon bounding box height, [xmin, xmax]."""
        return np.diff(self.zlim)[0]

    @property
    def box_area(self):
        """Return bounding box area."""
        return self.width*self.height

    @property
    def limit(self):
        """Return polygon bounding box (xmin, xmax, zmin, zmax)."""
        return self.xlim + self.zlim
