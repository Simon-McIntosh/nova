"""Generate polygons for CoilFrame instances."""
from dataclasses import dataclass

import shapely
import shapely.geometry
import shapely.ops
import numpy as np

from nova.geometry.polyshape import PolyShape


@dataclass
class PolyGen(PolyShape):
    """Manage shapely polygons."""

    def __call__(self, *args):
        """Evaluate poly."""
        return self._generate(*args)

    @property
    def _generate(self):
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
        return getattr(self, self.shape)

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
        buffer = point.buffer(radius, resolution=16)
        return shapely.geometry.Polygon(buffer)

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
        return shapely.geometry.Polygon(polygon)

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
        return shapely.geometry.Polygon(polygon)

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
        return shapely.geometry.Polygon(polygon)

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
        return shapely.geometry.Polygon(polygon)

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
        if height is None:
            length = width/2
        else:
            length = PolyGen.boxbound(width/2, height/np.sqrt(3))
        points = [[x_center + np.cos(alpha) * length,
                   z_center + np.sin(alpha) * length]
                  for alpha in np.linspace(0, 2*np.pi, 7)]
        return shapely.geometry.Polygon(points)


if __name__ == '__main__':

    poly = PolyGen('hex')
    print(poly(3, 4, 2))
