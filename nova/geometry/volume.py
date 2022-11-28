"""Volumetric methods for Vtkgeo class."""
from dataclasses import dataclass, field
import tempfile
from typing import ClassVar

import meshio
import numpy as np
import numpy.typing as npt
import pandas
import pyvista
import scipy.interpolate
from scipy.spatial.transform import Rotation
import shapely.geometry
import vedo
import vtk

from nova.geometry.polygeom import PolyGeom
from nova.geometry.polygon import Polygon
from nova.geometry.vtkgen import VtkFrame
from nova.geometry.line import Line


@dataclass
class TriShell:
    """Manage vtk shells."""

    mesh: vedo.Mesh
    qhull: bool = False
    ahull: bool = False
    features: ClassVar[list[str]] = [
        *'xyz', 'dx', 'dy', 'dz', 'dl', 'dt', 'area', 'volume']

    def __post_init__(self):
        """Create trimesh instance."""
        import trimesh
        self.tri = trimesh.Trimesh(self.mesh.points(), faces=self.mesh.faces())
        self._qhull = self._convex_hull

    @property
    def _convex_hull(self) -> vedo.Mesh:
        """Return decimated convex hull."""
        return vedo.ConvexHull(self.mesh.points()).decimate(
            n=6, method='pro', boundaries=True)

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
    def geom(self) -> list[float]:
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
        if self.ahull:
            try:
                from alphashape import alphashape
                from sklearn.cluster import DBSCAN
            except ImportError as error:
                raise ImportError('Generation of ahull poloidal polygons '
                                  'requires nova[\'mesh\']\n'
                                  'pip install nova[\'mesh\']') \
                    from error
            cluster = DBSCAN(eps=1e-3, min_samples=1)
            cluster.fit(poloidal)
            labels = np.unique(cluster.labels_)
            keypoints = np.zeros((len(labels), 2))
            for i, label in enumerate(labels):
                keypoints[i, :] = np.mean(
                    poloidal[label == cluster.labels_, :], axis=0)
            hull = alphashape(keypoints, 2.5)
            try:
                return Polygon(hull, name='ahull')
            except NotImplementedError:
                pass
        return Polygon(shapely.geometry.MultiPoint(poloidal).convex_hull,
                       name='qhull')


@dataclass
class TetVol(TriShell):
    """Manage vtk volumes."""

    tet: pyvista.UnstructuredGrid = field(init=False)

    def __post_init__(self):
        """Initialize tripanel and load volume."""
        super().__post_init__()
        self.load_volume()

    def load_volume(self):
        """Compute volume from closed surface mesh."""
        import trimesh
        with tempfile.NamedTemporaryFile(suffix='.msh') as tmp:
            trimesh.interfaces.gmsh.to_volume(self.tri, file_name=tmp.name)
            msh = meshio.read(tmp.name)
        cells = msh.cells[0][1]
        n_cells = len(cells)
        cells = np.append(np.full((n_cells, 1), 4, int), cells, axis=1)
        celltypes = np.full(n_cells, vtk.VTK_TETRA, int)
        points = msh.points
        self.tet = pyvista.UnstructuredGrid(cells, celltypes, points)
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


class Patch(vedo.Mesh):
    """Construct surface patch."""

    def __init__(self, points: npt.ArrayLike):
        cells = np.arange(0, len(points)).reshape(1, -1)
        super().__init__((points, cells))


class VtkPoly(VtkFrame):
    """Construct boundary line mesh from shapely polygon."""

    def __init__(self, poly: shapely.geometry.Polygon, c=None, alpha=1):
        points = np.c_[poly.boundary.xy[0],
                       np.zeros(len(poly.boundary.coords)),
                       poly.boundary.xy[1]]
        lines = [list(range(len(points)))]
        poly = vedo.utils.buildPolyData(points, lines=lines)
        super().__init__(poly, c, alpha)


class Ring(VtkFrame):
    """Construct vtk volume by rotating boundary polygon about z-axis."""

    def __init__(self, poly: shapely.geometry.Polygon, c=None, alpha=1):
        try:
            mesh = VtkPoly(poly).extrude(zshift=0, rotation=360, res=60)
        except NotImplementedError:  # multipart boundary (skin)
            mesh = [VtkPoly(Polygon(np.array(boundary.xy).T).poly).extrude(
                    zshift=0, rotation=360, res=60)
                    for boundary in poly.boundary.geoms]
            mesh = vedo.merge(mesh)
        super().__init__(mesh, c, alpha)
        self.flat()


@dataclass
class Section:
    """Transform 2D sectional data."""

    points: npt.ArrayLike
    origin: npt.ArrayLike = field(
        init=False, default_factory=lambda: np.zeros(3, float))
    triad: npt.ArrayLike = field(
        init=False, default_factory=lambda: np.identity(3, float))
    mesh_array: list[VtkFrame] = field(init=False, default_factory=list)
    point_array: list[npt.ArrayLike] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Append inital section to mesh."""
        self.points = np.array(self.points, float)

    def __len__(self):
        """Return length of mesh."""
        return len(self.mesh_array)

    def append(self):
        """Generate mesh and append mesh to list."""
        self.point_array.append(self.points.tolist())
        self.mesh_array.append(
            VtkFrame([self.points, [range(len(self.points))]]).c(len(self)))

    @staticmethod
    def normalize(vector):
        """Return normalized vector."""
        return vector / np.linalg.norm(vector)

    def to_vector(self, vector: npt.ArrayLike, coord: int):
        """Rotate mesh to vector."""
        vector = self.normalize(vector)
        cross = np.cross(self.triad[coord], vector)
        dot = np.dot(self.triad[coord], vector)
        if np.linalg.norm(cross) == 0 and dot == -1:  # catch -pi rotation
            axis = np.cross(self.triad[coord], self.triad[coord][::-1])
            rotation = Rotation.from_rotvec(-np.pi*axis)
        else:
            v_cross = np.array([[0, -cross[2], cross[1]],
                                [cross[2], 0, -cross[0]],
                                [-cross[1], cross[0], 0]])
            Rmat = np.identity(3) + v_cross + \
                np.dot(v_cross, v_cross) / (1+dot)
            rotation = Rotation.from_matrix(Rmat)
        self.points -= self.origin
        self.points = rotation.apply(self.points)
        self.points += self.origin
        self.triad = rotation.apply(self.triad)

    def to_point(self, point):
        """Translate points to point and store mesh."""
        delta = np.array(point - self.origin, float)
        self.origin += delta
        self.points += delta

    def sweep(self, mesh: pyvista.PolyData):
        """Sweep section along path."""
        for i in range(mesh.n_points):
            self.to_point(mesh.points[i])
            self.to_vector(mesh['normal'][i], 0)
            self.to_vector(mesh['tangent'][i], 1)
            self.append()
        return self

    def plot(self):
        """Plot mesh instances."""
        vedo.show(*self.mesh_array)


@dataclass
class Path:
    """Manage 3D path sub-divisions."""

    points: npt.ArrayLike
    delta: float = 0.
    mesh: pyvista.PolyData = field(init=False)
    submesh: pyvista.PolyData = field(init=False)

    def __post_init__(self):
        """Calculate length parameters and initialize interpolator."""
        self.mesh = Line.from_points(self.points).mesh
        self.interpolate()

    @classmethod
    def from_points(cls, points: npt.ArrayLike, delta=0):
        """Return submesh calculated from points and delta."""
        return cls(points, delta).submesh

    def interpolate(self):
        """Interpolate mesh to submesh."""
        arc_length = self._arc_length()
        sub_points = scipy.interpolate.interp1d(
            self.mesh['arc_length'], self.points, axis=0)(arc_length)
        self.submesh = Line.from_points(sub_points).mesh

    def _arc_length(self) -> npt.ArrayLike:
        """Return updated sub-segment spacing parameter."""
        if self.delta == 0:
            return self.mesh['arc_length']
        if self.delta < 0:  # specify segment number
            return np.linspace(self.mesh['arc_length'][0],
                               self.mesh['arc_length'][-1], -self.delta+1)
        segment_number = int(1 + self.mesh['arc_length'][-1] / self.delta)
        return np.linspace(self.mesh['arc_length'][0],
                           self.mesh['arc_length'][-1], segment_number)

    def plot(self):
        """Plot mesh and submesh loops."""
        vedo.show(self.mesh, self.submesh)


class Cell(VtkFrame):
    """Build vtk cell from a list of sectional polygons."""

    def __init__(self, point_array, link=False):
        """Construct vtk instance for cell constructed from bounding polys."""
        assert all((len(array) == len(point_array[0])
                    for array in point_array[1:]))
        point_array = np.array(point_array)
        if np.isclose(point_array[0, 0], point_array[0, -1]).all():
            point_array = point_array[:, :-1]  # open closed loop
        n_section, n_cap = point_array.shape[:2]
        points = np.vstack(point_array)
        nodes = np.arange(0, len(points)).reshape(-1, n_cap)
        cells = []
        for i in range(n_section-1):
            cells.extend(self._link(nodes[i], nodes[i+1]))
        if link:  # link start and end sections
            cells.extend(self._link(nodes[-1], nodes[0]))
        else:  # cap
            cells.append(nodes[0][::-1].tolist())  # base
            cells.append(nodes[-1].tolist())  # top
        super().__init__([points, cells])

    def _link(self, start, end):
        """Return list of rectangular cells linking start loop to end loop."""
        cells = np.zeros((len(start), 4), int)
        cells[:-1, 0] = start[:-1]
        cells[:-1, 1] = end[:-1]
        cells[:-1, 2] = end[1:]
        cells[:-1, 3] = start[1:]
        cells[-1, :] = [start[-1], end[-1], end[0], start[0]]
        cells = cells[:, ::-1]
        return cells.tolist()

    def __str__(self):
        """Return volume name."""
        return 'cell'


class Sweep(Cell):
    """Sweep polygon along path."""

    def __init__(self, poly, path, delta=0, link=False):
        """
        Sweep cross-section and return vedo.Mesh.

        Parameters
        ----------
        poly :
            - shapely.geometry.Polygon
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        path : npt.ArrayLike, shape(,3)
            Swept path.

        """
        if not isinstance(poly, Polygon):
            poly = PolyGeom(poly, segment='sweep')
        if isinstance(path, pyvista.PolyData):
            path = path.points
        mesh = Path.from_points(path, delta=delta)
        n_points = mesh.n_points
        if np.isclose(mesh.points[0], mesh.points[-1]).all():
            n_points -= 1
            link = True
        section = Section(poly.points).sweep(mesh)
        super().__init__(section.point_array, link=link)

    def __str__(self):
        """Return volume name."""
        return 'sweep'
