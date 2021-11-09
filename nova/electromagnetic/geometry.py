"""Geometric methods for FrameSpace class."""
from dataclasses import dataclass, field
from typing import ClassVar

import meshio
import numpy as np
import pandas
import pyvista as pv
from scipy.spatial.transform import Rotation
import tempfile
import trimesh
import vedo
import vtk

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.polygeom import PolyGeom
from nova.electromagnetic.polygen import PolyFrame
from nova.electromagnetic.vtkgen import VtkFrame


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
    def geom(self) -> tuple[float]:
        """Return list of geometry values as specified in self.features."""
        center = self.center_mass
        rotate = self.rotate
        extent = self.extent(rotate)
        rotvec = rotate.as_rotvec()
        area = self.volume / extent[2]
        return [*center, *rotvec, extent[0], extent[2], area, self.volume]

    @property
    def frame(self):
        """Return pannel Series."""
        return pandas.Series(self.geom, index=self.features)


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
class VtkGeo(MetaMethod):
    """Volume vtk geometry."""

    name = 'vtkgeo'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['vtk'])
    additional: list[str] = field(
        default_factory=lambda: [*TriPanel.features, 'body'])
    features: list[str] = field(
        init=False, default_factory=lambda: TriPanel.features)
    qhull: ClassVar[list[str]] = ['panel']
    geom: ClassVar[list[str]] = ['panel', 'stl']

    def initialize(self):
        """Init vtk panel data."""
        index = self.frame.index[~self.frame.geotype('Geo', 'vtk') &
                                 ~self.frame.geotype('Json', 'vtk') &
                                 ~pandas.isna(self.frame.vtk != '')]
        if (index_length := len(index)) > 0:
            frame = self.frame.loc[index, :]
            for i in range(index_length):
                tri = TriPanel(frame.vtk[i], qhull=frame.body[i] in self.qhull)
                frame.loc[frame.index[i], 'vtk'] = \
                    VtkFrame(tri.vtk.points(), tri.vtk.cells(),
                             c=tri.vtk.c(), alpha=tri.vtk.opacity())
                if frame.body[i] in self.geom:
                    frame.loc[frame.index[i], self.features] = tri.geom
                else:
                    frame.loc[frame.index[i], 'volume'] = tri.volume
            self.frame.loc[index, :] = frame


@dataclass
class PolyGeo(MetaMethod):
    """
    Polygon geometrical methods for FrameSpace.

    Extract geometric features from shapely polygons.
    """

    name = 'polygeo'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: [
        'segment', 'section', 'poly'])
    additional: list[str] = field(default_factory=lambda: [
        'dl', 'dt', 'rms', 'area'])
    require_all: bool = field(init=False, repr=False, default=False)
    base: list[str] = field(init=False, default_factory=lambda: [
        'x', 'y', 'z', 'segment', 'dx', 'dy', 'dz'])
    features: list[str] = field(init=False, default_factory=lambda: [
        'x', 'y', 'z', 'dx', 'dy', 'dz', 'area', 'rms'])

    def initialize(self):
        """Init sectional polygon data."""
        index = self.frame.index[~self.frame.geotype('Geo', 'poly') &
                                 (self.frame.segment != '') &
                                 (self.frame.section != '')]
        if (index_length := len(index)) > 0:
            section = self.frame.loc[index, 'section'].values
            coords = self.frame.loc[
                index, ['x', 'y', 'z', 'dx', 'dy', 'dz',
                        'segment', 'dl', 'dt']].to_numpy()
            poly = self.frame.loc[index, 'poly'].values
            poly_update = self.frame.loc[index, 'poly'].isna()
            geom = np.empty((index_length, len(self.features)), dtype=float)
            # itterate over index - generate poly as required
            for i in range(index_length):
                polygeom = PolyGeom(poly[i], *coords[i], section[i])
                section[i] = polygeom.section  # inflate section name
                if poly_update[i]:
                    poly[i] = PolyFrame(polygeom.poly, polygeom.section)
                geometry = polygeom.geometry  # extract geometrical features
                geom[i] = [geometry[feature] for feature in self.features]
            if poly_update.any():
                self.frame.loc[index, 'poly'] = poly
            self.frame.loc[index, self.features] = geom
            self.frame.loc[index, 'section'] = section

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.frame.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit
