"""Generate polygons for CoilFrame instances."""
import shapely
import shapely.geometry
import shapely.ops
import numpy as np
import string

polyshape = \
    dict.fromkeys(['circle', 'c', 'o'], 'circle') | \
    dict.fromkeys(['ellipse', 'elp', 'e'], 'ellipse') | \
    dict.fromkeys(['square', 'sq', 's'], 'square') | \
    dict.fromkeys(['rectangle', 'r'], 'rectangle') | \
    dict.fromkeys(['skin', 'sk'], 'skin') | \
    dict.fromkeys(['polygon', 'poly'], 'polygon') |\
    dict.fromkeys(['shell', 'shl', 'sh'], 'shell') |\
    dict.fromkeys(['hexagon', 'hex', 'h'], 'hexagon')


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


def circle(x_center, z_center, width, height=None):
    """
    Return shapely.cirle.

    Parameters
    ----------
    x_center : float
        Circle center, x-coordinate.
    z_center : float
        Circle center, z-coordinate.
    width : float
        Circle bounding box, x-dimension.
    height : Optional[float]
        Circle bounding box, z-dimension..

    Returns
    -------
    shape : shapely.polygon

    """
    diameter = boxbound(width, height)
    radius = diameter / 2
    point = shapely.geometry.Point(x_center, z_center)
    buffer = point.buffer(radius, resolution=32)
    return polyframe(shapely.geometry.Polygon(buffer.exterior), 'circle')


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
    return shapely.affinity.scale(circle(x_center, z_center, width),
                                  1, height/width)


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
    width = boxbound(width, height)
    return shapely.geometry.box(x_center-width/2, z_center-width/2,
                                x_center+width/2, z_center+width/2)


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
    return shapely.geometry.box(x_center-width/2, z_center-height/2,
                                x_center+width/2, z_center+height/2)


def skin(x_center, z_center, diameter, fill):
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
    fill : float
        Fill, 1-r/R. Must be greater than 0 and less than 1.
        Use circle for fill=1.

    Raises
    ------
    ValueError
        fill outside range 0-1.

    Returns
    -------
    shape : shapely.polygon

    """
    if fill < 0 or fill > 1:
        raise ValueError('skin fill not 0 <= '
                         f'{fill} <= 1')
    circle_outer = circle(x_center, z_center, diameter)
    if fill == 1:
        shape = circle_outer
    else:
        if fill == 0:
            fill = 1e-3
        scale = 1-fill
        circle_inner = shapely.affinity.scale(circle_outer, scale, scale)
        shape = circle_outer.difference(circle_inner)
    return shape


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
    length = boxbound(width/2, height/np.sqrt(3))
    points = [[x_center + np.cos(alpha) * length,
               z_center + np.sin(alpha) * length]
              for alpha in np.linspace(0, 2*np.pi, 7)]
    return shapely.geometry.Polygon(points)


class Polygon(shapely.geometry.Polygon):
    """Extend Polygon.__str__ for compact DataFrame representation."""

    def __init__(self, shell=None, holes=None, name='poly'):
        self.name = name
        super().__init__(shell, holes)

    def __str__(self):
        """Return polygon name."""
        return '-'


def polyframe(polygon, name='poly'):
    """Return polygon with compact __str__."""
    return Polygon(polygon.exterior, polygon.interiors, name)


def polygen(section):
    """
    Return shapely polygon.

    Parameters
    ----------
    section : str
        Required cross-section.

    Raises
    ------
    IndexError
        Cross-section not in [circle, ellipse, square, rectangle, skin].

    Returns
    -------
    shape : shapely.polygon

    """
    section = section.rstrip(string.digits)
    if polyshape[section] == 'circle':
        return circle
    if polyshape[section] == 'ellipse':
        return ellipse
    if polyshape[section] == 'square':
        return square
    if polyshape[section] == 'rectangle':
        return rectangle
    if polyshape[section] == 'skin':
        return skin
    if polyshape[section] == 'hexagon':
        return hexagon
    raise IndexError(f'cross_section: {section} not implemented'
                     f'\n specify as {polyshape}')


if __name__ == '__main__':

    poly = polygen('circle')
    print(poly(3, 4, 2))
