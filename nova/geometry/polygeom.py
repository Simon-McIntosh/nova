"""Manage PolyFrame geometrical data."""
from collections import namedtuple
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union

import numpy as np
import numpy.typing as npt
import shapely.geometry

from nova.geometry.polygon import PolyFrame
from nova.geometry.polygen import PolyGen
from nova.geometry.polygon import Polygon


@dataclass
class PolyGeom(Polygon):
    """Extract geometrical features from PolyFrame.

    Parameters
    ----------
    poly :
        - PolyFrame, shapely.geometry.Polygon
        - dict[str, list[float]], polyname: *args
        - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
        - array-like, shape(n,2) bounding loop [x, z]

    """

    poly: Union[Polygon, PolyFrame,
                shapely.geometry.Polygon, shapely.geometry.MultiPolygon,
                dict[str, list[float]],
                list[float, float, float, float],
                npt.ArrayLike] = field(repr=False)
    segment: str = 'ring'
    loop_length: float = 0
    name: str = field(init=False, default=None)

    def __post_init__(self):
        """Update loop length."""
        super().__post_init__()
        if np.isclose(self.loop_length, 0):
            self.loop_length = self.reference_length
            return
        if self.loop_length < 0:
            self.loop_length *= -self.reference_length
            return

    @cached_property
    def reference_length(self):
        """Return reference loop length."""
        if self.segment == 'ring':  # dy==ring circumference
            return 2*np.pi*self.centroid.x
        return 0

    @cached_property
    def centroid(self):
        """
        Return segment centroid.

        Returns
        -------
        centroid : namedtuple
            Segment centroid [x, y, z] coordinates.

        """
        data = namedtuple('point', 'x y z')
        return data(self.metadata.get('x_centroid', self.poly.centroid.x),
                    self.metadata.get('y_centroid', 0),
                    self.metadata.get('z_centroid', self.poly.centroid.y))

    @cached_property
    def delta(self):
        """
        Return geometry characteristic dimensions.

        Returns
        -------
        delta : namedtuple
            Segment deltas [x, y, z].

        """
        data = namedtuple('length', 'x y z')
        box = self.box
        return data(box.x, box.y, box.z)

    @property
    def length(self):
        """Return section characteristic length."""
        return self.metadata.get('length', None)

    @property
    def thickness(self):
        """Return section characteristic thickness."""
        return self.metadata.get('thickness', None)

    @cached_property
    def area(self):
        """Return polygon area."""
        if self.section == 'disc':
            return np.pi * self.length**2 / 4  # disc
        if self.section == 'square':
            return self.length**2  # square
        if self.section == 'rectangle':
            return self.length * self.thickness  # rectangle
        if self.section == 'skin':  # thickness = 1-r/R
            return np.pi * self.length**2 * self.thickness / 4 * \
                (2 - self.thickness**2)
        if self.section == 'hexagon':
            return 3/2 * self.width**2 / np.sqrt(3)
        return self.poly.area

    @cached_property
    def box(self):
        """Return width and height of polygon bounding box."""
        data = namedtuple('delta', 'x y z')
        if self.section in PolyGen.polyshape and self.section != 'skin' and \
                self.length is not None and \
                self.thickness is not None:
            if self.section in ['disc', 'square']:
                length = PolyGen.boxbound(self.length, self.thickness)
                return data(length, self.loop_length, length)
            return data(self.length, self.loop_length, self.thickness)
        return data(self.width, self.loop_length, self.height)

    @cached_property
    def rms(self):
        """
        Return section root mean square radius.

        Perform fast rms calculation for defined sections.
        Numerical calculation exicuted on section is not a base geometory.

        Parameters
        ----------
        section : str
            Cross section descriptor.
        centroid_radius : float
            Radial coordinate of geometric centroid.
        length : float
            First characteristic dimension, dl.
        thickness : float
            Second characteristic dimension, dt.
        poly : shapely.polygon, optional
            Polygon for numerical calculation if not in
            [disc, square, rectangle, skin]. The default is None.

        Returns
        -------
        radius : float
            Root mean square radius (uniform current density current center).

        """
        if self.segment != 'ring':
            return -1
        centroid_radius = self.centroid.x
        if self.section == 'disc':
            return np.sqrt(centroid_radius**2 + self.length**2 / 16)  # disc
        if self.section in ['square', 'rectangle']:
            return np.sqrt(centroid_radius**2 + self.length**2 / 12)  # square
        if self.section == 'skin':
            return np.sqrt((self.length**2 * self.thickness**2 / 24
                            - self.length**2 * self.thickness / 8
                            + self.length**2 / 8 + centroid_radius**2))
        return (shapely.ops.transform(
            lambda x, z: (x**2, z), self.poly).centroid.x)**0.5

    @cached_property
    def polydata(self):
        """Return polygon metadata."""
        centroid = self.centroid
        return dict(x_centroid=centroid.x, z_centroid=centroid.z,
                    length=self.length, thickness=self.thickness)

    @cached_property
    def geometry(self) -> dict[str, float]:
        """Return geometrical features."""
        centroid = self.centroid
        return {'x': centroid.x, 'y': centroid.y, 'z': centroid.z,
                'dl': self.length, 'dt': self.thickness,
                'dx': self.delta.x, 'dy': self.delta.y, 'dz': self.delta.z,
                'area': self.area, 'rms': self.rms,
                'poly': PolyFrame(self.poly, self.metadata),
                'section': self.section}


if __name__ == '__main__':

    geom = PolyGeom(Polygon({'hex': [3, 2, 0.1]})).geometry
