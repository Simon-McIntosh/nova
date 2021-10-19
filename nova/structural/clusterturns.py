"""Manage as-designed coil winding pack descriptors."""
from dataclasses import dataclass, field

import numpy as np
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
        self.update(self.n_clusters)

    def update(self, n_clusters):
        """Update turn clustering."""
        self.n_clusters = n_clusters
        self.extract_clusters()
        self.build_mesh()

    def extract_clusters(self):
        """Extract clusters from low-field mid-plane section."""
        n_turns = self.ccl_mesh.n_cells // 18
        self.turns = self.ccl_mesh['turns'][:n_turns]
        self.clusters = sklearn.cluster.KMeans(
            n_clusters=self.n_clusters,
            random_state=6, n_init=20, tol=1e-4).fit_predict(self.turns)

    def build_mesh(self):
        """Generate reduced ordered mesh from clustered turn data."""
        self.mesh = pv.PolyData()
        self.mesh.field_data.update(self.ccl_mesh.field_data)
        for i in range(18):
            coil = self.ccl_mesh.extract_cells(range(i*134, (i+1)*134))
            points = coil.points.reshape(134, -1, 3)
            point_data = {name: coil[name].reshape(134, -1, 3)
                          for name in coil.point_data if
                          len(coil[name].shape) == 2}
            for cluster in range(self.n_clusters):
                index = self.clusters == cluster
                cell = pv.Spline(points[index].mean(axis=0))
                arc_length = cell['arc_length']
                cell.clear_point_data()
                cell['arc_length'] = arc_length
                for name in point_data:
                    cell[name] = point_data[name][index].mean(axis=0)
                cell.cell_data['nturn'] = np.sum(index)
                cell.cell_data['coil'] = i
                cell.cell_data['cluster'] = cluster
                self.mesh += cell

    def plot_slice(self, axes=None, ms=8):
        """Plot clusters."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        for i in range(self.n_clusters):
            index = self.clusters == i
            color = f'C{i%10}'
            axes.plot(self.turns[index, 0], self.turns[index, 1],
                      's', color=color, ms=ms, alpha=0.5)
            axes.plot(self.turns[index, 0], self.turns[index, 1],
                      'o', ms=0.7*ms, color=color, alpha=0.5)
            axes.plot(self.turns[index, 0].mean(),
                      self.turns[index, 1].mean(), '.', color='gray')
            axes.axis('equal')
            axes.axis('off')
        axes.text(self.turns[:, 0].mean(), self.turns[:, 1].max(),
                  f'{self.n_clusters}', va='bottom')

    def plot_slice_array(self, clusters=[1, 5, 10]):
        """Plot low-filed mid-plane slices illustrating cluster options."""
        axes = plt.subplots(1, len(clusters))[1]
        for i, n_cluster in enumerate(clusters):
            self.update(n_cluster)
            self.plot_slice(axes[i])

    def plot_turn_array(self, clusters=[1, 3, 6, 11], nrows=2):
        """Plot reduced order turns."""
        ncols = int(np.ceil(len(clusters) / nrows))
        plotter = pv.Plotter(shape=(nrows, ncols))
        for i, n_cluster in enumerate(clusters):
            plotter.subplot(i//nrows, i % ncols)
            self.update(n_cluster)
            plotter.add_mesh(self.mesh)
        plotter.link_views()
        plotter.show()


if __name__ == '__main__':

    ccl = UniformWindingPack()
    cluster = ClusterTurns(ccl.mesh, 1)
    cluster.plot_slice()

    #print(cluster.mesh['nturn'])
    #cluster.update(1)
    cluster.plot_slice_array()

    #cluster.plot_turn_array()



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
