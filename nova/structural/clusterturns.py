"""Manage as-designed coil winding pack descriptors."""
from dataclasses import dataclass, field

import numpy.typing as npt
import pyvista as pv
import sklearn.cluster

from nova.structural.uniformwindingpack import UniformWindingPack
from nova.utilities.pyplot import plt


@dataclass
class ClusterTurns:
    """Cluster winding pack turns at low-field mid-plane."""

    ccl_mesh: pv.PolyData
    n_clusters: int = 1
    turns: npt.ArrayLike = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Identify clusters and build mesh."""
        self.extract()
        self.build()

    def extract(self):
        """Extract clusters from low-field mid-plane section."""
        n_turns = self.ccl_mesh.n_cells // 18
        self.turns = self.ccl_mesh['turns'][:n_turns]
        self.clusters = sklearn.cluster.KMeans(
            n_clusters=self.n_clusters,
            random_state=0).fit_predict(self.turns)

    def build(self):
        """Generate reduced ordered mesh from clustered turn data."""
        self.mesh = pv.PolyData()
        for i in range(18):
            coil = self.ccl_mesh.extract_cells(range(i*134, (i+1)*134))
            points = coil.points.reshape(134, -1, 3)
            for i in range(self.n_clusters):
                index = self.clusters == i
                self.mesh += pv.Spline(points[index].mean(axis=0))

    def plot(self):
        """Plot clusters."""
        for i in range(self.n_clusters):
            index = self.clusters == i
            plt.plot(self.turns[index, 0], self.turns[index, 1], 'o', ms=15)
            plt.axis('equal')
            plt.axis('off')


if __name__ == '__main__':

    ccl = UniformWindingPack()
    cluster = ClusterTurns(ccl.mesh, 5)
    cluster.plot()


    '''
        loops = self.mesh.points.reshape(134, -1, 3)
        mesh = pv.PolyData()
        for i in range(n_clusters):
            mesh += pv.Spline(loops[clusters == i].mean(axis=0))
        mesh.plot()

    def plot(self):
        """Plot coil current centerlines."""
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh)
        plotter.show()
    '''
