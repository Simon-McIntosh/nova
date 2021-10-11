from dataclasses import dataclass, field
import os

import numpy as np
import open3d as o3d
from pyobb.obb import OBB
import pyvista as pv
import scipy.spatial

from nova.definitions import root_dir
from nova.utilities.time import clock

@dataclass
class Shield:

    file: str = 'IWS_S6_BLOCKS'
    path: str = None
    mesh: pv.PolyData = field(init=False, repr=False)
    box: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        if self.path is None:
            self.path = os.path.join(root_dir, 'input/geometry/ITER/shield')
        self.load_mesh()
        self.load_box()

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

    def load_box(self):
        """Load orientated bounding boxes."""
        self.box = pv.PolyData()
        self.mesh = pv.PolyData()
        multiblockmesh = self.load_multiblock()
        n_mesh = len(multiblockmesh)
        tick = clock(n_mesh, header='loading orientated bounding boxes')
        for mesh in multiblockmesh:
            points = o3d.utility.Vector3dVector(mesh.points)
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(points)
            factor = (mesh.volume / np.prod(obb.extent))**(1 / 3)
            extent = factor*obb.extent
            bounds = np.zeros(6)
            bounds[::2] = -extent/2
            bounds[1::2] = extent/2
            box = pv.Box(bounds)
            box.points = box.points @ np.linalg.inv(obb.R)
            box.points += obb.center
            self.box += box
            self.mesh += mesh
            tick.tock()

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
        plotter.add_mesh(self.mesh, color='r')
        plotter.add_mesh(self.box, color='b', opacity=0.5)
        plotter.show()


if __name__ == '__main__':

    shield = Shield('IWS_S6_BLOCKS')
    shield.plot()
    #shield.load_stl()
