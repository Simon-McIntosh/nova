"""Generate polygons for CoilFrame instances."""
import shapely
import shapely.geometry
import numpy as np


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
        return super().__str__().split()[0]


def polyframe(polygon):
    """Return polygon with compact __str__."""
    return Polygon(polygon.exterior, polygon.interiors)


def polygen(cross_section):
    """
    Return shapely polygon.

    Parameters
    ----------
    cross_section : str
        Required cross-section.

    Raises
    ------
    IndexError
        Cross-section not in [circle, ellipse, square, rectangle, skin].

    Returns
    -------
    shape : shapely.polygon

    """
    if cross_section == 'circle':
        return circle
    if cross_section == 'ellipse':
        return ellipse
    if cross_section == 'square':
        return square
    if cross_section == 'rectangle':
        return rectangle
    if cross_section == 'skin':
        return skin
    raise IndexError(f'cross_section: {cross_section} not implemented'
                     '\n specify as '
                     '[circle, ellipse, square, rectangle, skin]')


def root_mean_square(cross_section, x_center, length, thickness, polygon=None):
    """
    Return section root mean square radius.

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
        Root mean square radius (current center for uniform current density).

    """
    if cross_section == 'circle':
        return np.sqrt(x_center**2 + length**2 / 16)  # circle
    if cross_section in ['square', 'rectangle']:
        return np.sqrt(x_center**2 + length**2 / 12)  # square
    if cross_section == 'skin':
        return np.sqrt((length**2 * thickness**2 / 24
                        - length**2 * thickness / 8
                        + length**2 / 8 + x_center**2))
    if polygon is None:
        polygon = polygen(cross_section)(x_center, 0, length, thickness)
    return (shapely.ops.transform(
        lambda x, z: (x**2, z), polygon).centroid.x)**0.5


if __name__ == '__main__':

    poly = polygen('circle')
    print(poly(3, 4, 2))
