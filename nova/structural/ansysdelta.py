from dataclasses import dataclass

import numpy as np
import pyvista as pv

from nova.structural.ansyspost import AnsysPost
from nova.structural.plotter import Plotter


@dataclass
class AnsysDelta(Plotter):
    target: str = "ccl0"
    baseline: str = "k0"
    part: str = "WP"
    folder: str = "TFCgapsG10"

    def __post_init__(self):
        """Calculate solution delta."""
        target_mesh = AnsysPost(self.folder, self.target, self.part).mesh
        baseline_mesh = AnsysPost(self.folder, self.baseline, self.part).mesh

        sort = np.argsort(target_mesh["ids"])
        unsort = np.zeros(target_mesh.n_points, dtype=int)
        unsort[sort] = np.arange(target_mesh.n_points)
        index = np.argsort(baseline_mesh["ids"])[unsort]

        self.mesh = pv.UnstructuredGrid()
        self.mesh.copy_structure(target_mesh)
        self.mesh.field_data.update(target_mesh.field_data)

        self.mesh["delta"] = target_mesh.points - baseline_mesh.points[index]

        for array in [
            array
            for array in target_mesh.point_data
            if array != "ids" and array in baseline_mesh.point_data
        ]:
            self.mesh[array] = target_mesh[array] - baseline_mesh[array][index]

    def plot(self):
        plotter = pv.Plotter()
        self.target.points = self.mesh.points
        plotter.add_mesh(self.mesh, color="r")
        plotter.add_mesh(self.target, color="b")
        plotter.show()


if __name__ == "__main__":
    delta = AnsysDelta()
    # delta.plot()

    # delta.mesh = delta.mesh.slice(normal=[0, 0, 1])
    delta.warp(500, opacity=0, displace="disp-1")
    # delta.animate('ccl0_cooldown', 'disp-1', 150)
