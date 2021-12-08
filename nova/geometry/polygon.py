"""Manage single instance polygon data."""
from dataclasses import dataclass, field
from typing import Union

import pygeos
import numpy as np
import numpy.typing as npt
import shapely.geometry

from nova.geometry.polyframe import PolyFrame
from nova.geometry.polygen import PolyGen


@dataclass
class Polygon:
    """Generate bounding polygon."""

    polyframe: Union[PolyFrame, pygeos.Geometry, shapely.geometry.Polygon,
                     dict[str, list[float]],
                     list[float, float, float, float],
                     npt.ArrayLike] = field(repr=False)
    name: str = None

    def __post_init__(self):
        """Process input geometry."""
        self.correct_aspect()
        metadata = self.extract()
        if self.name is not None:
            metadata |= dict(name=self.name)
        self.name = metadata['name']
        self.polyframe = PolyFrame(self.translate(), metadata)

    def __getattr__(self, attr):
        """Expose polyframe attribute."""
        return getattr(self.polyframe, attr)

    def correct_aspect(self):
        """Correct bounds to equal aspect geometries."""
        if isinstance(self.polyframe, dict):
            for section in self.polyframe:
                if PolyGen(section).shape in ['square', 'disc']:
                    if len(self.polyframe[section]) == 4:
                        length = PolyGen.boxbound(
                            *self.polyframe[section][-2:])
                        self.polyframe[section] = \
                            tuple(self.polyframe[section][:2]) + (length,)

    def extract(self) -> dict:
        """Return metadata extracted from input polygon."""
        if isinstance(self.polyframe, (Polygon, PolyFrame)):
            return self.metadata
        if isinstance(self.polyframe, (pygeos.Geometry, shapely.Geometry)):
            return dict(name='poly')
        if isinstance(self.polyframe, dict):
            metadata = dict(names=[PolyGen(name).shape
                                   for name in self.polyframe])
            if len(self.polyframe) == 1:
                metadata['name'] = metadata['names'][0]
                metadata |= {attr: value for attr, value in zip(
                    ['x_centroid', 'z_centroid', 'length', 'thickness'],
                    self.polyframe[next(iter(self.polyframe))])}
                metadata['section'] = metadata['name']
                return metadata
            metadata['name'] = '-'.join(metadata['names'])
            return metadata
        loop = np.array(self.polyframe)
        if loop.ndim == 1 and len(loop) == 4:  # bounding box
            metadata = self.bounding_box(*loop)
            metadata['section'] = metadata['name']
            return metadata
        return dict(name='polyloop')

    def translate(self):
        """Translate input geometry to shapely.geometry.Polygon.

        Parameters
        ----------
        poly :
            - PolyFrame, pygeos.Geometry, shapely.geometry.Polygon
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        Raises
        ------
        IndexError
            Malformed bounding box, shape is not (4,).
            Malformed bounding loop, shape is not (n, 2).

        Stores
        ------
        polygon : shapely.geometry.Polygon

        """
        if isinstance(self.polyframe, Polygon):
            return self.poly
        if isinstance(self.polyframe, (PolyFrame,
                                       pygeos.Geometry, shapely.Geometry)):
            return self.polyframe
        if isinstance(self.polyframe, dict):
            names = list(self.polyframe)
            polys = [PolyGen(section)(*self.polyframe[section])
                     for section in names]
            if len(polys) == 1:
                return polys[0]
            poly = shapely.ops.unary_union(polys)
            if not poly.is_valid:
                raise AttributeError('non-overlapping polygons specified in '
                                     f'{self.poly}')
            return poly
        loop = np.array(self.polyframe)  # to numpy array
        if loop.ndim == 1:   # poly bounding box
            if len(loop) == 4:  # [xmin, xmax, zmin, zmax]
                bbox = self.bounding_box(*loop)
                return PolyGen(bbox['name'])(*list(bbox.values())[1:])
            raise IndexError('malformed bounding box\n'
                             f'loop: {loop}\n'
                             'require [xmin, xmax, zmin, zmax]')
        if loop.shape[1] != 2:
            loop = loop.T
        if loop.ndim == 2 and loop.shape[1] == 2:  # loop
            return shapely.polygons(shapely.linearrings(loop))
        raise IndexError('malformed bounding loop\n'
                         f'shape(loop): {loop.shape}\n'
                         'require (n,2)')

    @staticmethod
    def bounding_box(xmin, xmax, zmin, zmax) -> dict:
        """Return characteristic dimensions of bounding box."""
        xlim, zlim = [xmin, xmax], [zmin, zmax]
        x_centroid = np.mean(xlim)
        z_centroid = np.mean(zlim)
        length = np.diff(xlim)[0]
        thickness = np.diff(zlim)[0]
        if np.isclose(length, thickness):
            name = 'square'
        else:
            name = 'rectangle'
        return dict(name=name, x_centroid=x_centroid, z_centroid=z_centroid,
                    length=length, thickness=thickness)
