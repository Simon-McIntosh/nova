"""Manage as-designed coil winding pack descriptors."""
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import sklearn.cluster

from nova.structural.uniformwindingpack import UniformWindingPack
from nova.utilities.pyplot import plt


@dataclass
class ClusterTurns:

    n_clusters: int = 1

    def __post_init__(self):
        self.windingpack = UniformWindingPack()
        self.points = self.windingpack.slice_coil(0, 0).points
        self.clusters = sklearn.cluster.KMeans(
            n_clusters=self.n_clusters,
            random_state=0).fit_predict(self.points)

    def build(self):
        ccl = self.windingpack.mesh

        self.mesh = pv.PolyData()
        for i in range(18):
            coil = ccl.extract_cells(range(i*134, (i+1)*134))
            points = coil.points.reshape(134, -1, 3)
            for i in range(self.n_clusters):
                index = self.clusters == i
                self.mesh += pv.Spline(points[index].mean(axis=0))
        self.mesh.plot()

    def plot(self):
        """Plot clusters."""
        for i in range(self.n_clusters):
            index = self.clusters == i
            plt.plot(self.points[index, 0], self.points[index, 1], '.')
            plt.axis('equal')


if __name__ == '__main__':

    cluster = ClusterTurns(5)
    cluster.build()


    '''

    '''

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
