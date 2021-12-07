"""Manage single instance polygon data."""
from dataclasses import dataclass, field
from typing import Union

import pandas
import pygeos
import numpy as np
import numpy.typing as npt
import shapely.geometry

from nova.geometry.polygen import PolyGen#, PolyFrame
from nova.utilities.pyplot import plt


@dataclass
class Polygon:
    """Generate bounding polygon."""

    poly: Union[dict[str, list[float]], npt.ArrayLike,
                shapely.geometry.Polygon] = field(repr=False)
    _hash: int = field(init=False, default=0)

    def __post_init__(self):
        """Generate bounding polygon."""
        self._generate()
        self._hash = hash(self.poly)
        #super().__post_init__()  # update geometric parameters

    def __hash__(self):
        """Return poly hash."""
        return id(self.poly)

    def _generate(self):
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
        if isinstance(self.poly, (pygeos.Geometry, shapely.Geometry)):
            return
        if isinstance(self.poly, dict):
            polys = [PolyGen(section)(*self.poly[section])
                     for section in self.poly]
            if len(polys) == 1:
                '''
                self.section = polys[0].name
                values = poly[next(iter(poly))]
                for attr, value in zip(['x_centroid', 'z_centroid',
                                        'length', 'thickness'], values):
                    setattr(self, attr, value)
                '''
                self.poly = polys[0]
                return
            self.poly = shapely.ops.unary_union(polys)
            if not self.poly.is_valid:
                raise AttributeError('non-overlapping polygons specified in '
                                     f'{self.poly}')
            return
        loop = np.array(self.poly)  # to numpy array
        if loop.ndim == 1:   # poly bounding box
            if len(loop) == 4:  # [xmin, xmax, zmin, zmax]
                xlim, zlim = loop[:2], loop[2:]
                centroid = np.mean(xlim), np.mean(zlim)
                length = np.diff(xlim)[0]
                thickness = np.diff(zlim)[0]
                if np.isclose(length, thickness):
                    section = 'square'
                else:
                    section = 'rectangle'
                self.poly = PolyGen(section)(*centroid, length, thickness)
                return
            raise IndexError('malformed bounding box\n'
                             f'loop: {loop}\n'
                             'require [xmin, xmax, zmin, zmax]')
        if loop.shape[1] != 2:
            loop = loop.T
        if loop.ndim == 2 and loop.shape[1] == 2:  # loop
            self.poly = shapely.polygons(shapely.linearrings(loop))
            return
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

    @property
    def points(self):
        """Return polygon points."""
        boundary = self.poly.boundary.xy
        return np.c_[boundary[0], np.zeros(len(boundary[0])), boundary[1]]


@dataclass
class PolyGeom:
    """Extract section geometrical features from shapely polygons."""

    poly: shapely.geometry.Polygon = field(default=None, repr=False)
    geom: dict[str, float] = field(
        init=None, default_factory=lambda: {
            attr: None for attr in ['x', 'y', 'z', 'dx', 'dy', 'dz',
                                    'dl', 'dt', ]})
    length: float = None
    thickness: float = None
    section: str = 'rectangle'
    segment: str = 'ring'


    def __post_init__(self):
        """Generate polygon as required."""
        self.update_section()
        if self.segment == 'ring':
            self.generate_ring()

    def update_section(self):
        """Update section name. Extract from poly, inflate str if not found."""
        try:
            self.section = self.poly.name
        except AttributeError:
            if self.section is not None:  # inflate shorthand
                self.section = PolyGen.polyshape[self.section]

    def generate_ring(self):
        """Generate poloidal polygon for circular filaments."""
        if self.section not in PolyGen.polyshape:  # clear
            self._centroid['x'] = self._centroid['z'] = None
        if pandas.isna(self.poly):
            self.poly = PolyGen(self.section)(
                *self.centroid[::2], self.length, self.thickness)
        self._delta['x'] = self.bbox[0]
        self._delta['y'] = 2*np.pi*self.centroid[0]  # diameter
        self._delta['z'] = self.bbox[1]

    @property
    def centroid(self):
        """
        Return segment centroid.

        Returns
        -------
        centroid : list[float, float, float]
            Segment centroid [x, y, z] coordinates.

        """
        if self.segment == 'ring':  # centered toroildal ring
            if self._centroid['x'] is None:
                self._centroid['x'] = self.poly.centroid.x  # update x centroid
            if self._centroid['y'] is None:
                self._centroid['y'] = 0
            if self._centroid['z'] is None:
                self._centroid['z'] = self.poly.centroid.y  # update z centroid
        return self._centroid['x'], self._centroid['y'], self._centroid['z']

    @property
    def area(self):
        """Return polygon area."""
        if self.section == 'disc':
            return np.pi * self.length**2 / 4  # disc
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
            if self.section in ['disc', 'square']:
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
            [disc, square, rectangle, skin]. The default is None.

        Returns
        -------
        radius : float
            Root mean square radius (uniform current density current center).

        """
        if self.segment != 'ring':
            return -1
        centroid_radius = self.centroid[0]
        if self.section == 'disc':
            return np.sqrt(centroid_radius**2 + self.length**2 / 16)  # disc
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
                'dx': self._delta['x'], 'dy': self._delta['y'], 'dz': self._delta['z'],
                'area': self.area, 'rms': self.rms,
                'poly': self.poly, 'section': self.section}
