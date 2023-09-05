"""Manage access to non-axisymmetric coil data."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from tqdm import tqdm

from nova.geometry.polygeom import Polygon
from nova.graphics.plot import Plot
from nova.imas.coil import coil_name
from nova.imas.database import ImasIds
from nova.imas.machine import CoilDatabase


@dataclass
class Elements(Plot):
    """Manage conductor element data extracted from coils_non_axisymmetric IDS."""

    points: np.ndarray = field(
        init=False, repr=False, default_factory=lambda: np.array([])
    )
    elements: ImasIds = field(repr=False, default=None)
    data: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    point_attrs: ClassVar[list[str]] = [
        "start_points",
        "intermediate_points",
        "end_points",
    ]

    def __post_init__(self):
        """Extract point data from elements node."""
        self.data["types"] = self.elements.types
        for attr in self.point_attrs:
            self.data[attr] = self._to_array(getattr(self.elements, attr))
        self.points = self._extract_points()

    def __len__(self):
        """Return maximum length of arrays stored in data dict."""
        return np.max([len(array) for array in self.data.values()])

    def _to_array(self, ids_points):
        """Return cartesian point array from nested cylindrical ids structure."""
        return np.c_[
            ids_points.r * np.cos(ids_points.phi),
            ids_points.r * np.sin(ids_points.phi),
            ids_points.z,
        ]

    @cached_property
    def _point_array(self):
        """Return shape (n, 3, 3) point array."""
        points = np.zeros((len(self), 3, 3))
        for i, attr in enumerate(self.point_attrs):
            if len(self.data[attr]) == 0:  # skip unfilled entries
                continue
            points[..., i] = self.data[attr]
        return points

    @cached_property
    def _type_array(self):
        """Return shape (n,) element types array."""
        types = np.zeros(len(self))
        types[:] = self.data["types"]
        return types

    def _extract_points(self):
        """Return shape (n, 3) unique point centerline."""
        points = []
        for element_type, point_array in zip(self._type_array, self._point_array):
            match element_type:
                case 1:
                    points.append(point_array[..., 0])
                case 2:
                    points.append(point_array[..., 0])
                    points.append(point_array[..., 1])
                case _:
                    raise ValueError(f"element_type {element_type} not supported")
        points.append(point_array[..., -1])
        return np.array(points)[
            np.sort(np.unique(points, return_index=True, axis=0)[1])
        ]

    def plot(self):
        """Plot polyline."""
        self.set_axes("3d")
        self.axes.plot(*self.points.T)


@dataclass
class Coils_Non_Axisymmetyric(Plot, CoilDatabase):
    """Manage access to coils_non_axisymmetric ids."""

    pulse: int = 115001
    run: int = 1
    name: str = "coils_non_axisymmetric"
    ids_node: str = "coil"
    vtk: bool = True

    coil_attrs: ClassVar[list[str]] = ["turns", "resistance"]

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        for coil in tqdm(self.ids_data.coil, "building coils non-axysymmetric"):
            for conductor in coil.conductor:
                elements = Elements(elements=conductor.elements)
                cross_section = Polygon(
                    np.c_[
                        conductor.cross_section.delta_r, conductor.cross_section.delta_z
                    ]
                )
                self.winding.insert(
                    cross_section, elements.points, label=coil_name(coil)
                )
        self.store()


if __name__ == "__main__":
    coil = Coils_Non_Axisymmetyric()
    coil.frame.vtkplot()
    # coil._clear()
