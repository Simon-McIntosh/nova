"""
Fast point-in-polygon algorithum.

code adapted from https://github.com/sasamil/PointInPolygon_Py/pointInside.
"""
from numba import njit
import numba
import numpy as np


@njit
def polypoint(polygon, point) -> bool:
    """
    Return boolean for point in polygon (is_inside_sm).

    Parameters
    ----------
    polygon: npt.ArrayLike, shape(:, 2)
        Bounding polygon, must form closed loop.

    point: npt.ArrayLike, shape(2)
        Point coordinates.

    Returns
    -------
    status: int
        0 - the point is outside the polygon
        1 - the point is inside the polygon
        2 - the point is one edge (boundary)


    This function gives the answer whether the given point is inside or
    outside the predefined polygon
    Unlike standard ray-casting algorithm, this one works on edges!
    (with no performance cost)
    According to performance tests - this is the best variant.

    """
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1
    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely
        # above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0]
                              or point[0] >= polygon[jj][0]):
            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + \
                    polygon[ii][0]
                # if line is left from the point - the ray moving towards left,
                # will intersect it
                if point[0] > F:
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2
            # point on upper peak (dy2=dx2=0)
            # or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and \
                (point[0] == polygon[jj][0] or
                 (dy == 0 and
                  (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0]) <= 0)):
                return 2
        ii = jj
        jj += 1
    return intersections & 1


@njit(parallel=True)
def polymultipoint(points, polygon):
    """Polygon membership status for multiple points (is_inside_sm_parallel).

    Parameters
    ----------
    polygon: npt.ArrayLike, shape(:, 2)
        Bounding polygon, must form closed loop.

    points: npt.ArrayLike, shape(:, 2)
        Point coordinates.

    Returns
    -------
    status: npt.ArrayLike[bool]
        0 - the point is outside the polygon
        1 - the point is inside the polygon
        2 - the point is one edge (boundary)

    Return boolean for point in polygon (originally - is_inside_sm).

    """
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = polypoint(polygon, points[i])
    return D
