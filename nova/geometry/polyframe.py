"""Manage single instance polygon data as polyframe."""

from dataclasses import dataclass, field
from functools import cached_property

import json
import numpy as np
import shapely.geometry

from nova.graphics.plot import Plot
from nova.geometry.geoframe import GeoFrame


@dataclass
class PolyFrame(Plot, GeoFrame):
    """Geometry object for dataframe polygons."""

    poly: shapely.geometry.Polygon
    name: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Update instance name."""
        super().__post_init__()
        if self.name is None:
            self.name = self.metadata.get("name", None)
        else:
            self.metadata["name"] = self.name

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
        metadata = polygon.pop("metadata", dict())
        return cls(shapely.geometry.shape(polygon), metadata)

    @property
    def section(self):
        """Return polygon section."""
        return self.metadata.get("section", self.name)

    @cached_property
    def centroid(self):
        """Return polygon centroid."""
        return self.poly.centroid

    @cached_property
    def area(self):
        """Return polygon area."""
        return self.poly.area

    def plot_boundary(self, axes=None, **kwargs):
        """Plot polygon boundary."""
        self.get_axes(axes=axes)
        self.axes.plot(*self.poly.exterior.xy, **kwargs)

    @cached_property
    def xlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.poly.bounds[::2]

    @cached_property
    def width(self) -> float:
        """Return polygon bounding box width."""
        return np.diff(self.xlim)[0]

    @cached_property
    def zlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.poly.bounds[1::2]

    @cached_property
    def height(self) -> float:
        """Return polygon bounding box height, [xmin, xmax]."""
        return np.diff(self.zlim)[0]

    @cached_property
    def box_area(self):
        """Return bounding box area."""
        if np.isclose((area := self.width * self.height), 0):
            return self.area
        return area

    @cached_property
    def limit(self):
        """Return polygon bounding box (xmin, xmax, zmin, zmax)."""
        return self.xlim + self.zlim

    @cached_property
    def points(self):
        """Return polygon points."""
        boundary = self.poly.boundary.xy
        return np.c_[boundary[0], np.zeros(len(boundary[0])), boundary[1]]

    @cached_property
    def boundary(self):
        """Return polygon boundary."""
        boundary = self.poly.boundary.xy
        return np.c_[boundary[0], boundary[1]]

    @cached_property
    def polygons(self):
        """Return polygon(s) coordinates a triple nested list."""
        try:
            geoms = self.poly.geoms
        except AttributeError:
            geoms = [self.poly]
        polys = [[poly.exterior, *poly.interiors] for poly in geoms]
        return (
            [[loop.xy[0].tolist() for loop in loops] for loops in polys],
            [[loop.xy[1].tolist() for loop in loops] for loops in polys],
        )
