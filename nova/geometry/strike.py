"""Manage strike point extraction."""
from dataclasses import dataclass, field

import numpy as np
from shapely import intersects, intersection
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

from nova.frame.baseplot import Plot
from nova.imas.database import Ids
from nova.imas.machine import Geometry


@dataclass
class Strike(Plot):
    """Extract strike-point locations."""

    wall: Ids | bool | str = 'iter_md'
    indices: tuple[int] = (1,)
    limiter: MultiLineString = field(default_factory=MultiLineString)
    contour: MultiLineString = field(default_factory=MultiLineString)
    point: Point | MultiPoint | None = field(init=False, default=None)
    intersects: bool = field(init=False, default=False)

    def __post_init__(self):
        """Extract limiter segments."""
        geometry = Geometry(False, False, wall=self.wall)
        wall = geometry['wall'](**geometry.wall, dplasma=-1)
        self.wall = wall.ids_attrs
        segments = [wall.segment(index) for index in self.indices]
        self.limiter = MultiLineString(segments)
        super().__post_init__()

    def update(self, lines):
        """Update intersections between contour and limiter surfaces."""
        self.contour = MultiLineString(lines)
        self.intersects = intersects(self.limiter, self.contour)
        if self.intersects:
            self.point = intersection(self.limiter, self.contour, 0)
        else:
            self.point = None
        return self

    @property
    def points(self):
        """Return radially sorted strike-point array."""
        if not self.intersects:
            return np.array([])
        point_array = np.c_[[[geom.x, geom.y] for geom in self['point']]]
        return point_array[np.argsort(point_array[:, 0])]

    def __getitem__(self, attr):
        """Return geometry itterable."""
        geometry = getattr(self, attr)
        if hasattr(geometry, 'geoms'):
            return geometry.geoms
        return [geometry]

    def plot(self):
        """Plot limiter, contour, and strike-point geometries."""
        self.set_axes('2d')
        for attr, style in zip(['limiter', 'contour', 'point'],
                               ['C0-', 'C7-', 'C3+']):
            for geom in self[attr]:
                self.axes.plot(*geom.xy, style)
        self.axes.set_xlim([*self.limiter.bounds[::2]])
        self.axes.set_ylim([*self.limiter.bounds[1::2]])


if __name__ == '__main__':

    strike = Strike(indices=(1,)).update([np.array([(3.5, -3), (7, -3)])])
    strike.plot()
    print(strike.points)
    #
