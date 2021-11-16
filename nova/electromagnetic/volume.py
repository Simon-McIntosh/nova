"""Volumetric methods for Vtkgeo class."""
from dataclasses import dataclass, field
import tempfile
from typing import ClassVar

import alphashape
import meshio
import numpy as np
import pandas
import pyvista as pv
import sklearn.cluster
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPolygon
import trimesh
import vedo
import vtk

from nova.electromagnetic.polygen import PolyFrame


@dataclass
class TriPanel:
    """Manage panel shells."""

    mesh: vedo.Mesh
    qhull: bool = False
    tri: trimesh.Trimesh = field(init=False, repr=False)
    features: ClassVar[list[str]] = [
        *'xyz', 'dx', 'dy', 'dz', 'dl', 'dt', 'area', 'volume']

    def __post_init__(self):
        """Create trimesh instance."""
        self.tri = trimesh.Trimesh(self.mesh.points(),
                                   faces=self.mesh.faces())
        self._qhull = self._convex_hull

    @property
    def _convex_hull(self) -> vedo.Mesh:
        """Return decimated convex hull."""
        return vedo.ConvexHull(self.mesh.points()).decimate(
            N=6, method='pro', boundaries=True)

    @property
    def panel(self) -> vedo.Mesh:
        """Return scaled convex hull."""
        mesh = self._qhull.clone()
        mesh.opacity(self.mesh.opacity())
        mesh.c(self.mesh.c())
        mesh.origin(*self.center_mass)
        mesh.scale((self.volume / mesh.volume())**(1/3))
        return mesh

    @property
    def vtk(self) -> vedo.Mesh:
        """Return vtk representation."""
        if self.qhull:
            return self.panel
        return self.mesh

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
        points = self._qhull.points()
        triangles = np.array(self._qhull.cells())
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
        points = self.rotate.inv().apply(self._qhull.points())
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        extent *= (self.volume / np.prod(extent))**(1 / 3)
        return extent

    @property
    def rotvec(self):
        """Return oriented bounding box rotation vector."""
        return self.rotate.as_rotvec()

    @property
    def geom(self) -> tuple[float]:
        """Return list of geometry values as specified in self.features."""
        center = self.center_mass
        rotate = self.rotate
        extent = self.extent(rotate)
        bounds = self.mesh.bounds()
        delta = np.array(bounds[1::2]) - np.array(bounds[::2])
        area = self.volume / extent[2]
        return [*center, *delta, extent[0], extent[2], area, self.volume]

    @property
    def frame(self):
        """Return pannel Series."""
        return pandas.Series(self.geom, index=self.features)

    @property
    def poly(self):
        """Return polodial polygon."""
        points = self.vtk.points()
        poloidal = np.zeros((len(points), 2))
        poloidal[:, 0] = np.linalg.norm(points[:, :2], axis=1)
        poloidal[:, 1] = points[:, 2]

        cluster = sklearn.cluster.DBSCAN(eps=1e-3, min_samples=1)
        cluster.fit(poloidal)
        labels = np.unique(cluster.labels_)
        keypoints = np.zeros((len(labels), 2))
        for i, label in enumerate(labels):
            keypoints[i, :] = np.mean(poloidal[label == cluster.labels_, :],
                                      axis=0)
        hull = alphashape.alphashape(keypoints, 2.5)
        try:
            return PolyFrame(hull, name='vtk')
        except NotImplementedError:
            return PolyFrame(hull.convex_hull, name='vtk')


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
