"""Manage single instance polygon data as polyframe."""
from dataclasses import dataclass, field

import json
import numpy as np
import shapely.geometry

from nova.geometry.geoframe import GeoFrame
from nova.plot import plt


@dataclass
class PolyFrame(GeoFrame):
    """Geometry object for dataframe polygons."""

    poly: shapely.geometry.Polygon
    metadata: dict = field(default_factory=dict)

    def __eq__(self, other) -> bool:
        """Return result of comparison with other."""
        if isinstance(other, PolyFrame):
            return self.poly == other.poly
        return self.poly == other

    def dumps(self) -> str:
        """Return geojson representation."""
        data = self.poly.__geo_interface__ | dict(metadata=self.metadata)
        return json.dumps(data)

    @classmethod
    def loads(cls, data: str):
        """Load geojson prepresentation."""
        polygon = json.loads(data)
        metadata = polygon.pop('metadata', dict())
        return cls(shapely.geometry.shape(polygon), metadata)

    @property
    def name(self):
        """Return polygon name."""
        return self.metadata.get('name', 'polyframe')

    @property
    def section(self):
        """Return polygon section."""
        return self.metadata.get('section', self.name)

    @property
    def centroid(self):
        """Return polygon centroid."""
        return self.poly.centroid

    @property
    def area(self):
        """Return polygon area."""
        return self.poly.area

    def plot_boundary(self, axes=None, **kwargs):
        """Plot polygon boundary."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        axes.plot(*self.poly.exterior.xy, **kwargs)

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

    @property
    def boundary(self):
        """Return polygon boundary."""
        boundary = self.poly.boundary.xy
        return np.c_[boundary[0], boundary[1]]

    # def orient(self):
    #     """Return coerced polygon boundary as a positively oriented curve."""
    #     self.poly = shapely.geometry.polygon.orient(self.poly)
