"""Manage polygon creation."""
from dataclasses import dataclass, field
from typing import Union

import shapely.geometry
import shapely.strtree
import numpy as np

from nova.electromagnetic.polygen import polygen, PolyFrame
from nova.electromagnetic.polygeom import PolyGeom
from nova.utilities.pyplot import plt

# pylint:disable=unsubscriptable-object


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
            polys = [polygen(section)(*poly[section]) for section in poly]
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
                return polygen(self.section)(self.x_centroid, self.z_centroid,
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

    def validate(self):
        """Repair polygon if not valid."""
        '''
        if not polygon.is_valid:
            polygon = pygeos.creation.polygons(loop)
            polygon = pygeos.constructive.make_valid(polygon)
            area = [pygeos.area(pygeos.get_geometry(polygon, i))
                    for i in range(pygeos.get_num_geometries(polygon))]
            polygon = pygeos.get_geometry(polygon, np.argmax(area))
            polygon = shapely.geometry.Polygon(
                pygeos.get_coordinates(polygon))
        '''

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
