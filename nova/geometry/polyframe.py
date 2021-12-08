"""Manage single instance polygon data as polyframe."""
from dataclasses import dataclass, field

import geojson
import json
import numpy as np
import shapely.geometry

from nova.geometry.geoframe import GeoFrame
from nova.utilities.pyplot import plt


@dataclass
class PolyFrame(GeoFrame):
    """Geometry object for dataframe polygons."""

    poly: shapely.geometry.Polygon
    metadata: dict = field(default_factory=dict)

    @property
    def name(self):
        """Return polygon name."""
        return self.metadata.get('name', 'polyframe')

    def __eq__(self, other) -> bool:
        """Return result of comparison with other."""
        if isinstance(other, PolyFrame):
            return self.poly == other.poly
        return self.poly == other

    def __getattr__(self, attr):
        """Return shapely polygon attributes."""
        if hasattr(self.poly, attr):
            return getattr(self.poly, attr)

    def dumps(self) -> str:
        """Return geojson representation."""
        data = dict(poly=geojson.dumps(self.poly), metadata=self.metadata)
        return json.dumps(data)

    @classmethod
    def loads(cls, data: str):
        """Load geojson prepresentation."""
        data = json.loads(data)
        return cls(shapely.geometry.shape(data['poly'], data['metadata']))

    def plot_boundary(self):
        """Plot polygon boundary."""
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

    @property
    def points(self):
        """Return polygon points."""
        boundary = self.poly.boundary.xy
        return np.c_[boundary[0], np.zeros(len(boundary[0])), boundary[1]]

    # def orient(self):
    #     """Return coerced polygon boundary as a positively oriented curve."""
    #     self.poly = shapely.geometry.polygon.orient(self.poly)
