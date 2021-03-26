"""Generate polygons for CoilFrame instances."""
import shapely
import shapely.geometry
import shapely.ops
import numpy as np

polyshape = \
    dict.fromkeys(['circle', 'c', 'o'], 'circle') | \
    dict.fromkeys(['ellipse', 'e'], 'ellipse') | \
    dict.fromkeys(['square', 'sq'], 'square') | \
    dict.fromkeys(['rectangle', 'r'], 'rectangle') | \
    dict.fromkeys(['skin', 'sk'], 'skin')


def boundbox(width, height):
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
    diameter = boundbox(width, height)
    radius = diameter / 2
    point = shapely.geometry.Point(x_center, z_center)
    buffer = point.buffer(radius, resolution=32)
    return shapely.geometry.Polygon(buffer.exterior)


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
    width = boundbox(width, height)
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


def skin(x_center, z_center, diameter, fractional_thickness):
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
    fractional_thickness : float
        Fractional thickness, 1-r/R. Must be greater than 0 and less than 1.
        Use circle for fractional_thickness=1.

    Raises
    ------
    ValueError
        fractional_thickness outside range 0-1.

    Returns
    -------
    shape : shapely.polygon

    """
    if fractional_thickness < 0 or fractional_thickness > 1:
        raise ValueError('skin fractional thickness not 0 <= '
                         f'{fractional_thickness} <= 1')
    circle_outer = circle(x_center, z_center, diameter)
    if fractional_thickness == 1:
        shape = circle_outer
    else:
        if fractional_thickness == 0:
            fractional_thickness = 1e-3
        scale = 1-fractional_thickness
        circle_inner = shapely.affinity.scale(circle_outer, scale, scale)
        shape = circle_outer.difference(circle_inner)
    return shape


class Polygon(shapely.geometry.Polygon):
    """Extend Polygon.__str__ for compact DataFrame representation."""

    def __str__(self):
        """Return compact __str__."""
        return super().__str__().split()[0][:4]


def polyframe(polygon):
    """Return polygon with compact __str__."""
    return Polygon(polygon.exterior, polygon.interiors)


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
    raise IndexError(f'cross_section: {section} not implemented'
                     f'\n specify as {polyshape}')


if __name__ == '__main__':

    poly = polygen('circle')
    print(poly(3, 4, 2))
