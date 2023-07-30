"""Manage access to non-axisymmetric coil data."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from nova.graphics.plot import Plot
from nova.imas.coil import coil_name
from nova.imas.machine import CoilDatabase


@dataclass
class Coils_Non_Axisymmetyric(Plot, CoilDatabase):
    """Manage access to coils_non_axisymmetric ids."""

    pulse: int = 115001
    run: int = 1
    name: str = "coils_non_axisymmetric"
    ids_node: str = "coil"
    available: list[str] = field(default_factory=lambda: ["vtk"])

    coil_attrs: ClassVar[list[str]] = ["turns", "resistance"]

    def _points(self, ids_points):
        """Return cartesian point array from nested cylindrical ids structure."""
        return np.c_[
            ids_points.r * np.cos(ids_points.phi),
            ids_points.r * np.sin(ids_points.phi),
            ids_points.z,
        ]

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        for coil in self.ids_data.coil[:1]:
            coil_name(coil)
            points = []
            for conductor in coil.conductor:
                points.extend(
                    np.r_[
                        self._points(conductor.elements.start_points),
                        self._points(conductor.elements.intermediate_points),
                        self._points(conductor.elements.end_points),
                    ]
                )
            print(np.shape(points))

            self.points = points
            # line = Line().from_points(np.array(points))
            # line.show()
            # self.winding.insert({"c": [0, 0, 0.005]}, np.array(points), name=name)
            # polyline = PolyLine(np.array(points))
            # polyline.plot()

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
    coil.subframe.vtkplot()
    # coil._clear()
