"""Generate deformed geometries based on fiducial data."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from nova.assembly.fiducialdata import FiducialData
from nova.assembly.plotter import Plotter


@dataclass
class FiducialMesh(Plotter):
    """Manage fiducial mesh."""

    mesh: pv.PolyData = field(default_factory=pv.PolyData)
    data: FiducialData = field(init=False)

    def __post_init__(self):
        """Load fiducial database."""
        self.data = FiducialData(fill=True, sead=2025)

    def add(self, mesh):
        """Add geometry to fiducial mesh."""
        self.mesh = self.mesh.merge(mesh, merge_points=False)

    def clear_mesh(self):
        """Re-initialize mesh."""
        self.mesh = pv.PolyData()

    def load_coil(self, index):
        """Add centerline from fiducial data to mesh."""
        self.add(self.data.mesh.extract_cells(index))

    def load_coilset(self):
        """Load full TF coilset."""
        self.clear_mesh()
        for cell_index in range(self.data.dim["coil"]):
            self.load_coil(cell_index)


if __name__ == "__main__":
    fiducialmesh = FiducialMesh()
    fiducialmesh.load_coil(3)
    fiducialmesh.warp(500)

    # fiducial.load_coilset()
    fiducialmesh.clear_mesh()
    for index in range(18):
        fiducialmesh.load_coil(index)
    # for index in range(19):
    #    fiducialmesh.load_coil(index, clear_mesh=False)
    fiducialmesh.warp(500)
