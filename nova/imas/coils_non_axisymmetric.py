"""Manage access to non-axisymmetric coil data."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np

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
    available: list[str] = field(default_factory=lambda: ["vtk"])

    coil_attrs: ClassVar[list[str]] = ["turns", "resistance"]

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        for coil in self.ids_data.coil[:1]:
            print(coil_name(coil))
            for i, conductor in enumerate(coil.conductor):
                elements = Elements(elements=conductor.elements)

            print(len(elements.points))
            from nova.geometry.polyline import PolyLine

            polyline = PolyLine(elements.points)
            print(len(polyline.segments))
            self.winding.insert({"c": [0, 0, 0.05]}, elements.points)
            # elements.plot()
            # line = Line().from_points(np.array(points))
            # line.show()
            # self.winding.insert({"c": [0, 0, 0.005]}, np.array(points), name=name)

        # self.polyline = polyline
        # self.frame.vtk.plot()

        """
        coil_name = coil_names(self.ids_data.coil)
        with self.build_scenario():
            self.data.coords["point"] = ["x", "y", "z"]
            self.data.coords["coil_name"] = coil_name
            self.append("coil_name", self.coil_attrs)

            self.data["indices"] = "coil_name", np.zeros(
                self.data.dims["coil_name"], int
            )
            points = []
            for i, coil in enumerate(self.ids_data.coil):
                for conductor in coil.conductor:
                    points.extend(
                        np.r_[
                            self._points(conductor.elements.start_points),
                            self._points(conductor.elements.intermediate_points),
                            self._points(conductor.elements.end_points),
                        ]
                    )
                if i == 0:
                    self.data["indices"][i] = 0
                    continue
                self.data["indices"][i] = self.data["indices"][i - 1] + len(points[-1])
            print(np.shape(points))

            self.data["points"] = ("index", "point"), points
            # point_array = points.r * np.cos(points.phi)
            # y_coord = points.r * np.sin(points.phi)
            # z_coord = points.z

            # mesh.plot()
            # xarray.
            """


if __name__ == "__main__":
    coil = Coils_Non_Axisymmetyric()
    coil.frame.vtkplot()
    # coil._clear()
