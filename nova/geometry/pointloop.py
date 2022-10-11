"""
Fast point(s)-in-polygon algorithum.

code adapted from https://github.com/sasamil/PointInPolygon_Py/pointInside.
"""
import numba
import numpy as np


class PointLoop:
    """Point in loop methods."""

    def __init__(self, points, status=1):
        self.points = points
        self.status = status
        self.point_number = len(self.points)
        self.select = np.empty(self.point_number, dtype=bool)

    @staticmethod
    @numba.njit
    def point_in_polygon(point, polygon) -> int:
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
            2 - the point is on edge (boundary)

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
                    # if line is left from the point
                    # the ray moving towards left, will intersect it
                    if point[0] > F:
                        intersections += 1
                    elif point[0] == F:  # point on line
                        return 2
                # point on upper peak (dy2=dx2=0)
                # or horizontal line (dy=dy2=0 and dx*dx2<=0)
                elif dy2 == 0 and \
                    (point[0] == polygon[jj][0] or
                     (dy == 0 and
                      (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])
                      <= 0)):
                    return 2
            ii = jj
            jj += 1
        return intersections

    def update(self, polygon):
        """
        Update loop membership status for multiple points.

        Derived from is_inside_sm_parallel.

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
        for i in range(self.point_number):
            self.select[i] = \
                self.point_in_polygon(self.points[i], polygon) & self.status
        return self.select
