from dataclasses import dataclass, field
import os

import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy.spatial.transform import Rotation

import tetgen
import vedo

from nova.definitions import root_dir
from nova.utilities.time import clock


@dataclass
class Block:

    mesh: Union[pv.PolyData, vedo.Mesh]
    grid: int = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.mesh, vedo.Mesh):
            self.mesh = pv.PolyData(self.mesh.polydata())

    def tetrahedralize(self):
        tet = tetgen.TetGen(polydata)
        tet.tetrahedralize(order=1, quality=False)
        grid = tet.grid.compute_cell_sizes(length=False, area=False)


    @staticmethod
    def center(mesh: vedo.Mesh):
        """Return center of mass."""

        return np.sum(grid['Volume'].reshape(-1, 1) *
                      grid.cell_centers().points, axis=0) / grid.volume

    @staticmethod
    def rotate(mesh: vedo.Mesh):
        """Return PCA rotational transform."""
        mesh = mesh.fillHoles()
        points = mesh.points()
        triangles = np.array(mesh.cells())
        vertex = dict(a=points[triangles[:, 0]],
                      b=points[triangles[:, 1]],
                      c=points[triangles[:, 2]])
        normal = np.cross(vertex['b']-vertex['a'], vertex['c']-vertex['a'])
        l2norm = np.linalg.norm(normal, axis=1)
        covariance = np.cov(normal, rowvar=False, aweights=l2norm**5)
        eigen = np.linalg.eigh(covariance)[1]
        eigen /= np.linalg.det(eigen)
        return Rotation.from_matrix(eigen)

    @staticmethod
    def extent(mesh: vedo.Mesh, rotate: Rotation):
        """Return box extent."""
        points = rotate.inv().apply(mesh.points())
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        extent *= (mesh.volume() / np.prod(extent))**(1 / 3)
        return extent

    @staticmethod
    def box(center: npt.ArrayLike, extent: npt.ArrayLike, rotate: Rotation):
        """Return pannel bounding box."""
        bounds = np.zeros(6)
        bounds[::2] = -extent/2
        bounds[1::2] = extent/2
        box = pv.Box(bounds)
        box.points = rotate.apply(box.points)
        box.points += center
        return vedo.Mesh(box)

    @staticmethod
    def convex_hull(mesh: vedo.Mesh):
        """Return decimated convex hull."""
        return vedo.ConvexHull(mesh.points()).decimate(
                N=10, method='pro', boundaries=True)


@dataclass
class Shield:

    file: str = 'IWS_S6_BLOCKS'
    path: str = None
    mesh: pv.PolyData = field(init=False, repr=False)
    geom: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load datasets."""
        if self.path is None:
            self.path = os.path.join(root_dir, 'input/geometry/ITER/shield')
        self.load_mesh()
        self.load_frame()

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.path, f'{self.file}.vtk')

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

    def load_frame(self):
        """Retun multiblock mesh."""
        mesh = vedo.Mesh(self.vtk_file)
        parts = mesh.splitByConnectivity(1)

        blocks = []
        tick = clock(len(parts), header='loading decimated convex hulls')
        for part in parts:
            self.part = part
            part.cap()
            convex_hull = self.convex_hull(part)
            blocks.append(convex_hull.opacity(1).c('b'))
            '''
            part.cap()
            convex_hull = self.convex_hull(part)
            blocks.append(convex_hull.c('b').opacity(1))
            try:
                center = self.center(part)
            except RuntimeError:  # Failed to tetrahedralize (non-manifold)
                self.part = part.clone()
                center = self.center(part.decimate(0.1))
            #center = self.center(convex_hull)
            #center = part.centerOfMass()
            #rotate = self.rotate(m)
            #extent = self.extent(m, rotate)
            #box.append(self.box(center, extent, rotate))
            '''
            tick.tock()

        vedo.show(blocks)

    def plot(self):
        """Plot mesh."""
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, color='r', opacity=1)
        plotter.add_mesh(self.box, color='g', opacity=0.75, show_edges=True)

        #plotter.add_mesh(self.cell, color='b', opacity=1)
        plotter.show()


if __name__ == '__main__':

    shield = Shield('IWS_S6_BLOCKS')
    #shield.plot()
    #shield.load_stl()
