"""Generate deformed geometries based on fiducial data."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from nova.structural.fiducialdata import FiducialData
from nova.structural.plotter import Plotter


@dataclass
class FiducialMesh(Plotter):
    """Manage fiducial mesh."""

    mesh: pv.PolyData = field(default_factory=pv.PolyData)
    fiducial_data: FiducialData = field(init=False)

    def __post_init__(self):
        """Load fiducial database."""
        self.fiducial_data = FiducialData(fill=True, sead=2025)

    def add(self, mesh):
        """Add geometry to fiducial mesh."""
        if 'delta' not in mesh.array_names:
            mesh['delta'] = np.zeros((mesh.n_points, 3))
        try:
            delta = np.append(self.mesh['delta'], mesh['delta'], axis=0)
        except KeyError:
            delta = mesh['delta']
        self.mesh += mesh
        self.mesh['delta'] = delta

    def clear_mesh(self):
        """Re-initialize mesh."""
        self.mesh = pv.PolyData()

    def add_centerline(self, index):
        """Add centerline from fiducial data to mesh."""
        self.add(self.fiducial_data.extract_cells(index))

    def load_centerline(self, index):
        """Load single fiducial centerline."""
        self.clear_mesh()
        self.add_centerline(index)


if __name__ == '__main__':

    fiducial = FiducialMesh()
    fiducial.load_centerline(0)
    fiducial.warp(500)
