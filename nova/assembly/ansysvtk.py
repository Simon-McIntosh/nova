"""Load Ansys vtk data."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import pyvista as pv

from nova.assembly.ansyspost import AnsysPost


@dataclass
class AnsysVTK(AnsysPost):
    """Post-process Ansys output from F4E's 18TF coil model."""

    folder: str = "TFCgapsG10"
    file: str = None
    subset: str = "all"
    mesh: pv.UnstructuredGrid = field(init=False, repr=False)

    scenario_list: ClassVar[list[str]] = [
        "preload",
        "cooldown",
        "TFonly",
        "SOD",
        "SOP",
        "XPF",
        "CS1_0",
        "CS2U0",
        "CS2L0",
        "SOF",
        "SOB",
        "EOB",
        "EOP",
        "EOB+PD",
    ]

    def __post_init__(self):
        """Load ansys vtk data."""
        super().__post_init__()
        self.build_scenarios()

    def build_scenarios(self):
        """Build ansys scenarios."""
        ansys = AnsysPost(
            self.folder, self.file, self.subset, self.data_dir, self.rst_dir
        ).mesh
        ansys = self.mesh.copy()
        self.mesh.clear_point_data()
        for scn in self.scenario:
            try:
                self.mesh[scn] = ansys[f"disp-{self.scenario[scn]}"]
            except KeyError:
                pass
        self.mesh.field_data["scenario"] = list(self.scenario)

    @cached_property
    def scenario(self):
        """Return secenario lookup."""
        return {
            self.scenario_list[int(index - 1)]: i
            for i, index in enumerate(self.mesh.field_data["time_support"])
        }


if __name__ == "__main__":
    vtk = AnsysVTK(file="k2", subset="all")

    # vtk.mesh -= AnsysVTK(file='k1', subset='wp').mesh
    vtk.mesh["TFonly-cooldown"] = vtk.mesh["TFonly"] - vtk.mesh["cooldown"]
    # vtk.mesh = vtk.mesh.clip_box([-15, 15, -15, 15, -15, 0], invert=False)
    # vtk.mesh = vtk.mesh.clip_box([-5, 5, -5, 5, -0.5, 0], invert=False)
    vtk.warp(factor=120, opacity=0, displace="TFonly-cooldown", view="xy")

    # vtk.animate('k1', 'TFonly-cooldown', view='iso',
    #            max_factor=150, zoom=1.3, opacity=0)
