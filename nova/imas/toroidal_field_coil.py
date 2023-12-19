"""Manage access to non-axisymmetric coil data."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from tqdm import tqdm

from nova.geometry.polygeom import Polygon
from nova.graphics.plot import Plot
from nova.imas.coil import coil_name, coil_names
from nova.imas.coils_non_axisymmetric import Elements
from nova.imas.machine import CoilDatabase
from nova.imas.scenario import Scenario


@dataclass
class ToroidalFieldCoil(Plot, CoilDatabase, Scenario):
    """Manage access to coils_non_axisymmetric ids."""

    pulse: int = 111002
    run: int = 1
    name: str = "tf"
    ids_node: str = "coil"

    coil_attrs: ClassVar[list[str]] = ["turns", "resistance"]

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        name = coil_names(self.ids_data.coil)
        with self.build_scenario():
            self.data.coords["coil_name"] = name
            self.data.coords["coil_index"] = "coil_name", range(len(name))
            self.data.coords["point"] = ["x", "y", "z"]

            points = {}
            for coil in tqdm(self.ids_data.coil, "building toroidal field coils"):
                name = coil_name(coil)
                points[name] = []
                for i, conductor in enumerate(coil.conductor):
                    elements = Elements(elements=conductor.elements)
                    radius = np.linalg.norm(elements.points[0, :2])
                    polygon = Polygon(
                        np.c_[
                            -4 * radius * conductor.cross_section.delta_phi,
                            conductor.cross_section.delta_r,
                        ]
                    )
                    points[name].extend(elements.points)
                    if i > 0:
                        name = f"name{i}"
                    self.winding.insert(
                        polygon,
                        elements.points,
                        name=name,
                        part="tf",
                    )
                    if i > 0:
                        self.linkframe([coil_name(coil), name])
                points[name] = np.array(points[name])
            maximum_point_number = np.max([len(_points) for _points in points.values()])
            self.data.coords["points_index"] = np.arange(1, maximum_point_number + 1)
            self.data["points"] = ("coil_name", "points_index", "points"), np.zeros(
                (
                    self.data.sizes["coil_name"],
                    self.data.sizes["points_index"],
                    self.data.sizes["point"],
                )
            )
            self.data["points_length"] = "coil_name", [
                len(_points) for _points in points.values()
            ]
            for i, (number, name) in enumerate(
                zip(self.data["points_length"].data, self.data["coil_name"].data)
            ):
                self.data["points"].data[i, :number] = points[name]


if __name__ == "__main__":
    coil = ToroidalFieldCoil()

    coil._clear()
