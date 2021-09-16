"""Construct error displacement fields from pairs of TFC18 instances"""

from dataclasses import dataclass, field
import os

import pyvista as pv

from nova.definitions import root_dir
from nova.structural.datadir import DataDir
from nova.structural.plotter import Plotter
from nova.structural.TFC18 import TFC18


@dataclass
class TFCgap(DataDir, Plotter):
    """Construct error displacement fields for TF coilset."""

    cluster: int = None
    baseline: str = 'k0'
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
        target = model.mesh.copy()
        model.reload(self.baseline)
        baseline = model.mesh.copy()
        for scenario in model.mesh.field_data['scenario']:
            try:
                self.mesh[scenario] = target[scenario] - baseline[scenario]
            except KeyError:
                pass

    def animate(self, max_factor=200):
        """Animate displacement."""
        folder = os.path.join(self.directory, 'gif')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        filename = os.path.join(folder,
                                f'{self.loadcase[0]}_{self.loadcase[1]}_delta')
        super().animate(filename, 'delta', view='iso', max_factor=max_factor)


if __name__ == '__main__':

    gap = TFCgap('TFCgapsG10', 'ccl0', baseline='k0', cluster=False)

    #mesh = gap.mesh.slice(normal=[0, 0, 1])
    #clip = pv.Cylinder(direction=(0, 0, 1), radius=3)
    #gap.mesh = mesh.clip_surface(clip, invert=True)

    #p = pv.Plotter()
    #p.add_mesh(mesh)
    #p.show()

    gap.warp(100, displace='cooldown', opacity=0)
    #gap.animate(1000)
