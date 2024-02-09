"""Generate geodesic centerline from uniform winding-pack mesh."""

from dataclasses import dataclass, field
import os

import numpy as np
import pyvista as pv

from nova.definitions import root_dir
from nova.geometry.line import Line
from nova.assembly.uniformwindingpack import UniformWindingPack
from nova.assembly.clusterturns import ClusterTurns
import matplotlib.pyplot as plt


@dataclass
class CenterLine(Line):
    """Manage uniform winding-pack centerline."""

    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load coil centerline."""
        self.load()

    @property
    def vtk_file(self):
        """Return vtk file path (Centerline)."""
        return os.path.join(root_dir, "input/ITER/TF_CL.vtk")

    def load(self):
        """Load single coil centerline."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.build()
            self.mesh.save(self.vtk_file)

    def build(self):
        """Build single coil centerline from uniform winding-pack."""
        windingpack = UniformWindingPack()
        cluster = ClusterTurns(windingpack.mesh, 1)
        points = cluster.mesh.cell_points(0)
        self.mesh = pv.Spline(np.append(points, points[:1], axis=0))
        self.mesh["arc_length"] /= self.mesh["arc_length"][-1]
        self.compute_vectors()

    def plot(self, axes=None):
        """Plot 2D loop."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]
            plt.axis("equal")
            plt.axis("off")
        axes.plot(self.mesh.points[:, 0], self.mesh.points[:, 2])

    def plot_vectors(self, vector="normal"):
        """Plot centerline vectors as glyphs."""
        if vector not in self.mesh.array_names:
            raise IndexError(f"vector {vector} not in {self.mesh.array_names}")
        self.mesh[vector] = self.mesh[vector]
        self.mesh.active_vectors_name = vector
        self.mesh.arrows.plot(scalars="arc_length")


if __name__ == "__main__":
    cl = CenterLine()
    cl.mesh.plot()
