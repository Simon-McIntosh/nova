from dataclasses import dataclass, field
import os

import meshio
import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy.spatial.transform import Rotation
import tempfile
import trimesh
import vedo
import vtk

from nova.definitions import root_dir
from nova.electromagnetic.framespace import FrameSpace
from nova.utilities.time import clock


@dataclass
class TriPanel:
    """Manage panel shells."""

    mesh: vedo.Mesh
    tri: trimesh.Trimesh = field(init=False, repr=False)
    scale: float = 1e-3
    qhull: vedo.Mesh = field(init=False, repr=False)

    def __post_init__(self):
        """Create trimesh instance."""
        self.mesh = self.mesh.scale(self.scale)
        self.tri = trimesh.Trimesh(self.mesh.points(),
                                   faces=self.mesh.faces())
        self.qhull = self._convex_hull

    @property
    def _convex_hull(self) -> vedo.Mesh:
        """Return decimated convex hull."""
        return vedo.ConvexHull(self.mesh.points()).decimate(
            N=6, method='pro', boundaries=True)

    @property
    def panel(self) -> vedo.Mesh:
        """Return scaled convex hull."""
        mesh = self.qhull
        mesh.origin(*self.center_mass)
        mesh.scale((self.volume / mesh.volume())**(1/3))
        return mesh

    @property
    def volume(self):
        """Return grid volume."""
        return self.tri.volume

    @property
    def center_mass(self):
        """Return grid center of mass."""
        return self.tri.center_mass

    @property
    def rotate(self) -> Rotation:
        """Return PCA rotational transform."""
        points = self.qhull.points()
        triangles = np.array(self.qhull.cells())
        vertex = dict(a=points[triangles[:, 0]],
                      b=points[triangles[:, 1]],
                      c=points[triangles[:, 2]])
        normal = np.cross(vertex['b']-vertex['a'], vertex['c']-vertex['a'])
        l2norm = np.linalg.norm(normal, axis=1)
        covariance = np.cov(normal, rowvar=False, aweights=l2norm**5)
        eigen = np.linalg.eigh(covariance)[1]
        eigen /= np.linalg.det(eigen)
        return Rotation.from_matrix(eigen)

    def extent(self, rotate=None):
        """Return optimal bounding box extents."""
        if rotate is None:
            rotate = self.rotate
        points = self.rotate.inv().apply(self.qhull.points())
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        extent *= (self.volume / np.prod(extent))**(1 / 3)
        return extent

    @property
    def frame(self):
        """Return pannel dataframe [mm]."""
        center = self.center_mass
        rotate = self.rotate
        extent = self.extent(rotate)
        rotvec = rotate.as_rotvec()
        area = self.volume / extent[2]
        return dict(x=center[0], y=center[1], z=center[2],
                    dx=rotvec[0], dy=rotvec[1], dz=rotvec[2],
                    dl=extent[0], dt=extent[2], area=area,
                    volume=self.volume, poly=self.panel, section='')


@dataclass
class TetPanel(TriPanel):
    """Manage panel volumes."""

    tet: pv.UnstructuredGrid = field(init=False)

    def __post_init__(self):
        """Initialize tripanel and load volume."""
        super().__post_init__()
        self.load_volume()

    def load_volume(self):
        """Compute volume from closed surface mesh."""
        with tempfile.NamedTemporaryFile(suffix='.msh') as tmp:
            trimesh.interfaces.gmsh.to_volume(self.tri, file_name=tmp.name)
            msh = meshio.read(tmp.name)
        cells = msh.cells[0][1]
        n_cells = len(cells)
        cells = np.append(np.full((n_cells, 1), 4, int), cells, axis=1)
        celltypes = np.full(n_cells, vtk.VTK_TETRA, int)
        points = msh.points
        self.tet = pv.UnstructuredGrid(cells, celltypes, points)
        self.tet = self.tet.compute_cell_sizes(length=False, area=False)

    @property
    def cell_centers(self):
        """Return cell centers."""
        return self.tet.cell_centers().points

    @property
    def cell_volumes(self):
        """Return cell volumes."""
        return self.tet['Volume'].reshape(-1, 1)

    @property
    def volume(self):
        """Return grid volume."""
        return np.sum(self.cell_volumes)

    @property
    def center_mass(self):
        """Return grid center of mass."""
        return np.sum(self.cell_volumes*self.cell_centers,
                      axis=0) / self.volume


@dataclass
class Shield:
    """Manage shield sector."""

    file: str = 'IWS_FM_PLATE_S4'
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
            self.mesh = self.read_stl()

    def read_stl(self):
        """Read stl file."""
        mesh = pv.read(self.stl_file)
        mesh.save(self.vtk_file)
        return mesh

    def load_frame(self):
        """Retun multiblock mesh."""
        mesh = vedo.Mesh(self.vtk_file)
        parts = mesh.splitByConnectivity(10)

        frame = FrameSpace(required=['x', 'y', 'z'], label='fi',
                           segment='volume')
        #parts = [vedo.Mesh(pv.read('tmp.vtk'))]
        blocks = []

        tick = clock(len(parts), header='loading decimated convex hulls')
        for i, part in enumerate(parts):
            #pv.PolyData(part.polydata()).save('tmp.vtk')
            tri = TriPanel(part)

            frame += tri.frame

            #blocks.append(tri.mesh.opacity(1).c(i))
            blocks.append(tri.panel.opacity(1).c(i+1))
            #blocks.append(tri.panel.opacity(1).c(i+2))

            #tet = TetPanel(tri.panel)

            #print(tri.volume, tet.volume)
            #print(block.center_mass)
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
        self.frame = frame

        #vedo.show(frame.poly)

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
