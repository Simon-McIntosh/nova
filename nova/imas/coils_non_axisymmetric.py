"""Manage access to non-axisymmetric coil data."""
from dataclasses import dataclass, field

import numpy as np

from nova.imas.coil import coil_names
from nova.graphics.plot import Plot
from nova.imas.scenario import Scenario


@dataclass
class Coils_Non_Axisymmetyric(Plot, Scenario):
    """Manage access to coils_non_axisymmetric ids."""

    name: str = "coils_non_axisymmetric"
    ids_node: str = "coil"
    coil_attrs: list[str] = field(default_factory=lambda: ["turns", "resistance"])

    def _points(self, ids_points):
        """Return cartesian point array from nested cylindrical ids structure."""
        return np.c_[
            ids_points.r * np.cos(ids_points.phi),
            ids_points.r * np.sin(ids_points.phi),
            ids_points.z,
        ]

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
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
                    points.append(
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

            # self.data["points"] =
            # point_array = points.r * np.cos(points.phi)
            # y_coord = points.r * np.sin(points.phi)
            # z_coord = points.z

            # mesh.plot()
            # xarray.


if __name__ == "__main__":
    pulse, run = 115001, 1
    Coils_Non_Axisymmetyric(pulse, run, "iter_md")._clear()
    coil = Coils_Non_Axisymmetyric(pulse, run, "iter_md")
