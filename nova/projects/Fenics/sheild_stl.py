from dataclasses import dataclass, field
import os

import numpy as np
import open3d as o3d
import pyvista as pv
import sklearn.decomposition
import scipy.spatial
import trimesh
import vedo

from nova.definitions import root_dir
from nova.utilities.time import clock


@dataclass
class Shield:

    file: str = 'IWS_S6_BLOCKS'
    path: str = None
    mesh: pv.PolyData = field(init=False, repr=False)
    geom: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        if self.path is None:
            self.path = os.path.join(root_dir, 'input/geometry/ITER/shield')
        self.load_mesh()
        self.load_geom()

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.path, f'{self.file}.vtk')

    @property
    def vtmb_file(self):
        """Retun full vtmb filename."""
        return os.path.join(self.path, f'{self.file}.vtmb')

    @property
    def stl_file(self):
        """Return full stl filename."""
        return os.path.join(self.path, f'{self.file}.stl')

    def load_mesh(self):
        """Load mesh."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.mesh = self.load_vtk()

    def load_vtk(self):
        """Load vtk mesh from file."""
        mesh = pv.read(self.stl_file)
        mesh.save(self.vtk_file)
        return mesh

    def load_geom(self):
        """Extract convexhull for each pannel."""
        self.mesh = pv.PolyData()
        self.qhull = pv.PolyData()
        multiblockmesh = self.load_multiblock()
        n_mesh = len(multiblockmesh)
        n_mesh = 20
        tick = clock(n_mesh, header='loading orientated bounding boxes')

        center_of_mass = np.zeros((n_mesh, 3))
        volume = np.zeros(n_mesh)
        for i, mesh in enumerate(multiblockmesh[slice(0, n_mesh)]):
            center_of_mass[i] = mesh.center_of_mass()
            volume[i] = mesh.volume

            hull = scipy.spatial.ConvexHull(mesh.points)
            faces = np.column_stack((3*np.ones((len(hull.simplices), 1),
                                               dtype=int), hull.simplices))
            self.qhull += pv.PolyData(hull.points, faces.flatten())

            self.mesh += mesh
            tick.tock()
        self.cell = pv.PolyData(center_of_mass)
        self.cell['volume'] = volume

    def qhull(self):
        '''

        mesh = np.sum(mesh.delaunay_3d())

        points = mesh.cell_centers().points
        mesh = mesh.compute_cell_sizes(length=False, volume=False)
        covariance = np.cov(points, rowvar=False, ddof=None,
                            aweights=mesh['Area'])
        eigen_vectors = np.linalg.eigh(covariance)[1]

        points = points @ eigen_vectors
        extent = np.max(points, axis=0) - np.min(points, axis=0)

        bounds = np.zeros(6)
        bounds[::2] = -extent/2
        bounds[1::2] = extent/2
        box = pv.Box(bounds)
        box.points = box.points @ eigen_vectors.T
        box.points += mesh.center_of_mass()
        self.box += box

        npoints = len(hull.points)
        self.cell += mesh
        '''

    def load_multiblock(self):
        """Retun multiblock mesh."""
        try:
            return pv.read(self.vtmb_file)
        except FileNotFoundError:
            multiblockmesh = self.load_vtk()
            multiblockmesh = multiblockmesh.split_bodies(
                label=True, progress_bar=True)
            multiblockmesh.save(self.vtmb_file)
            return multiblockmesh

    def plot(self):
        """Plot mesh."""
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, color='r', opacity=1)
        #plotter.add_mesh(self.qhull, color='g', opacity=1)

        plotter.add_mesh(self.cell, color='b', opacity=1)
        plotter.show()


if __name__ == '__main__':

    shield = Shield('IWS_S6_BLOCKS')
    shield.plot()
    #shield.load_stl()
