"""Generate polygons for CoilFrame instances."""
from dataclasses import dataclass, field
import string
from typing import ClassVar

import geojson
import shapely
import shapely.geometry
import shapely.ops
import numpy as np

from nova.geometry.geoframe import GeoFrame


class PolyFrame(shapely.geometry.Polygon, GeoFrame):
    """Extend Polygon.__str__ for compact DataFrame representation."""

    def __init__(self, shell=None, holes=None, name='poly'):
        self.name = name
        self.polygon = None #shapely.geometry.Polygon(shell, holes)
        super().__init__(shell, holes)


    def __str__(self):
        """Return polyframe name."""
        return self.name

    '''
    def __getattr__(self, attr):
        """Return attrs from polygon else from self."""
        if hasattr(self.polygon, attr):
            return getattr(self.polygon, attr)
        return getattr(self, attr)
    '''

    def dumps(self) -> str:
        """Return geojson representation."""
        return geojson.dumps(self)

    @classmethod
    def loads(cls, poly: str):
        """Load geojson prepresentation."""
        return cls(shapely.geometry.shape(geojson.loads(poly)))


@dataclass
class PolyGen:
    """Manage shapely polygon."""

    section: str
    polyshape: ClassVar[dict[str, str]] = \
        dict.fromkeys(['disc', 'dsc', 'd', 'o', 'disk', 'dsk',
                       'circle', 'c'], 'disc') | \
        dict.fromkeys(['ellipse', 'ellip', 'el', 'e'], 'ellipse') | \
        dict.fromkeys(['square', 'sq', 's'], 'square') | \
        dict.fromkeys(['rectangle', 'rect', 'r'], 'rectangle') | \
        dict.fromkeys(['skin', 'sk'], 'skin') | \
        dict.fromkeys(['polygon', 'poly'], 'polygon') |\
        dict.fromkeys(['shell', 'shl', 'sh'], 'shell') |\
        dict.fromkeys(['hexagon', 'hex', 'hx', 'h'], 'hexagon')
    poly: PolyFrame = field(init=False)

    def __post_init__(self):
        """Generate polygon."""
        self.poly = self.generate_polygon()

    def __call__(self, *args):
        """Evaluate poly."""
        return self.poly(*args)

    def generate_polygon(self):
        """
        Generate shapley polygon from section name.

        Parameters
        ----------
        section : str
            Required cross-section.

        Raises
        ------
        IndexError
            Cross-section not in [disc, ellipse, square, rectangle, skin].

        Returns
        -------
        shape : shapely.polygon

        """
        section = self.section.rstrip(string.digits)
        if self.polyshape[section] == 'disc':
            return self.disc
        if self.polyshape[section] == 'ellipse':
            return self.ellipse
        if self.polyshape[section] == 'square':
            return self.square
        if self.polyshape[section] == 'rectangle':
            return self.rectangle
        if self.polyshape[section] == 'skin':
            return self.skin
        if self.polyshape[section] == 'hexagon':
            return self.hexagon
        raise IndexError(f'cross_section: {self.section} not implemented'
                         f'\n specify as {self.polyshape}')

    @staticmethod
    def boxbound(width, height):
        """
        Return minimum dimension.

        Parameters
        ----------
        width : float
            Horizontal dimension.
        height : Union[float, None]
            Vertical dimension.

        Returns
        -------
        mindim
            Dimension of minimum bounding box.

        """
        if height is None:
            return width
        return np.min([width, height])

    @staticmethod
    def disc(x_center, z_center, width, height=None):
        """
        Return shapely.cirle.

        Parameters
        ----------
        x_center : float
            Disc center, x-coordinate.
        z_center : float
            Disc center, z-coordinate.
        width : float
            Disc bounding box, x-dimension.
        height : Optional[float]
            Disc bounding box, z-dimension..

        Returns
        -------
        shape : shapely.polygon

        """
        diameter = PolyGen.boxbound(width, height)
        radius = diameter / 2
        point = shapely.geometry.Point(x_center, z_center)
        buffer = point.buffer(radius, resolution=64)
        return PolyFrame(buffer, 'disc')

    @staticmethod
    def ellipse(x_center, z_center, width, height):
        """
        Return shapely.ellipse.

        Parameters
        ----------
        x_center : float
            Ellipse center, x-coordinate.
        z_center : float
            Ellipse center, z-coordinate.
        width : float
            Ellipse width, x-dimension.
        height : float
            Ellipse height, z-dimension.

        Returns
        -------
        shape : shapely.polygon

        """
        polygon = shapely.affinity.scale(PolyGen.disc(
            x_center, z_center, width), 1, height/width)
        return PolyFrame(polygon, name='ellipse')

    @staticmethod
    def square(x_center, z_center, width, height=None):
        """
        Return shapely.square.

        Parameters
        ----------
        x_center : float
            Square center, x-coordinate.
        z_center : float
            Square center, z-coordinate.
        width : float
            Square width.
        height : Optional[float]
            Square height.

        Returns
        -------
        shape : shapely.polygon

        """
        width = PolyGen.boxbound(width, height)
        polygon = shapely.geometry.box(x_center-width/2, z_center-width/2,
                                       x_center+width/2, z_center+width/2)
        return PolyFrame(polygon, name='square')

    @staticmethod
    def rectangle(x_center, z_center, width, height):
        """
        Return shapely.rectangle.

        Parameters
        ----------
        x_center : float
            Rectangle center, x-coordinate.
        z_center : float
            Rectangle center, z-coordinate.
        width : float
            Rectangle width, x-dimension.
        height : float
            Rectangle height, z-dimension.

        Returns
        -------
        shape : shapely.polygon

        """
        polygon = shapely.geometry.box(x_center-width/2, z_center-height/2,
                                       x_center+width/2, z_center+height/2)
        return PolyFrame(polygon, name='rectangle')

    @staticmethod
    def skin(x_center, z_center, diameter, factor):
        """
        Return shapely.ring.

        Parameters
        ----------
        x_center : float
            Ring center, x-coordinate.
        z_center : float
            Ring center, z-coordinate.
        diameter : float
            External diameter.
        factor : float
            factor = 1-r/R. Must be greater than 0 and less than 1.
            Use disc for factor=1.

        Raises
        ------
        ValueError
            factor outside range 0-1.

        Returns
        -------
        shape : shapely.polygon

        """
        if factor <= 0 or factor > 1:
            raise ValueError('skin factor not 0 <= '
                             f'{factor} <= 1')
        disc_outer = PolyGen.disc(x_center, z_center, diameter)
        if factor == 1:
            return disc_outer
        scale = 1-factor
        disc_inner = PolyGen.disc(x_center, z_center, scale*diameter)
        polygon = disc_outer.difference(disc_inner)
        return PolyFrame(polygon, name='skin')

    @staticmethod
    def hexagon(x_center, z_center, width, height=None):
        """
        Return shapely.polygon.

        Parameters
        ----------
        x_center : float
            Hexagon center, x-coordinate.
        z_center : float
            Hexagon center, z-coordinate.
        width : float
            Hexagon width, x-dimension.
        height : Optional[float]
            Hexagon height, z-dimension.

        Returns
        -------
        shape : shapely.polygon

        """
        length = PolyGen.boxbound(width/2, height/np.sqrt(3))
        points = [[x_center + np.cos(alpha) * length,
                   z_center + np.sin(alpha) * length]
                  for alpha in np.linspace(0, 2*np.pi, 7)]
        return PolyFrame(points, name='hexagon')


if __name__ == '__main__':

    poly = PolyGen('disc')
    print(poly(3, 4, 2))
