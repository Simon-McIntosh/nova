"""Construct error displacement fields from pairs of TFC18 instances"""

from dataclasses import dataclass, field

import pyvista as pv

from nova.structural.datadir import DataDir
from nova.structural.plotter import Plotter
from nova.structural.TFC18 import TFC18


@dataclass
class TFCgap(DataDir, Plotter):
    """Construct error displacement fields for TF coilset."""

    folder: str = "TFCgapsG10"
    cluster: int = None
    baseline: str = "k0"
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load analysis data."""
        self.load_ansys_data()

    def load_ansys_data(self):
        """Diffrence model data (file - baseline)."""
        model = TFC18(*self.args, cluster=self.cluster)
        self.mesh = model.mesh.copy()
        self.mesh.clear_point_data()
        self.mesh.field_data.update(model.mesh.field_data)
        self.mesh["arc_length"] = model.mesh["arc_length"]
        target = model.mesh.copy()
        model.reload(self.baseline)
        baseline = model.mesh.copy()
        if "scenario" not in model.mesh.array_names:
            model.reload_mesh()
        for scenario in model.mesh.field_data["scenario"]:
            try:
                self.mesh[scenario] = target[scenario] - baseline[scenario]
            except KeyError:
                pass

    def animate(self, displace="EOB", max_factor=200):
        """Animate displacement."""
        filename = f"{self.file}_{self.baseline}_{displace}"
        super().animate(filename, displace, view="xy", max_factor=max_factor, opacity=0)


if __name__ == "__main__":
    gap = TFCgap("TFCgapsG10", "k1", baseline="k0", cluster=None)

    mesh = gap.mesh.slice(normal=[0, 0, 1])
    clip = pv.Cylinder(direction=(0, 0, 1), radius=3)
    gap.mesh = mesh.clip_surface(clip, invert=True)

    # p = pv.Plotter()
    # p.add_mesh(mesh)
    # p.show()

    gap.warp(160, displace="TFonly", opacity=0.5, view="xy")
    # gap.animate('EOB', 1000)
