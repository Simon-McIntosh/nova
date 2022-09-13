"""Generate grid and solution methods for hexagonal plasma filaments."""
from dataclasses import dataclass, field
from typing import Union, ClassVar

import numba
import numpy as np
import pyvista
import scipy.spatial
import xarray

from nova.biot.biotframe import BiotTarget
from nova.biot.biotgrid import BiotBaseGrid
from nova.biot.biotsolve import BiotSolve
from nova.electromagnetic.error import GridError
from nova.geometry.pointloop import PointLoop


@dataclass
class BiotPlasmaGrid(BiotBaseGrid):
    """Compute interaction across hexagonal grid."""

    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    levels: Union[int, list[float]] = 21

    def solve(self):
        """Solve Biot interaction across plasma grid."""
        if self.sloc['plasma'].sum() == 0:
            raise GridError('plasma')
        target = BiotTarget(self.subframe.loc['plasma', ['x', 'z']].to_dict())
        self.data = BiotSolve(self.subframe, target, reduce=[True, False],
                              attrs=self.attrs, chunks=self.chunks).data
        self.tessellate()
        super().post_solve()

    @staticmethod
    @numba.njit
    def loop_neighbour_vertices(points, neighbor_vertices, boundary_vertices):
        """Calculate 6-point ordered loop vertex indices."""
        point_number = len(points)
        neighbours = np.full((point_number, 6), -1)
        for i in range(len(points)):
            if i in boundary_vertices:
                continue
            center_point = points[i, :]
            slice_index = slice(neighbor_vertices[0][i],
                                neighbor_vertices[0][i+1])
            neighbour_index = neighbor_vertices[1][slice_index]
            if len(neighbour_index) != 6:
                continue
            delta = points[neighbour_index] - center_point
            angle = np.arctan2(delta[:, 1], delta[:, 0])
            neighbours[i] = neighbour_index[np.argsort(angle)[::-1]]
        mask = neighbours[:, 0] != -1
        stencil_index = np.arange(point_number)[mask]
        stencil = np.append(np.arange(point_number)[mask].reshape(-1, 1),
                            neighbours[mask], axis=1)
        return stencil, stencil_index

    def tessellate(self):
        """Tesselate hexagonal mesh, compute 6-point neighbour loops."""
        points = self.subframe.loc['plasma', ['x', 'z']].to_numpy()
        tri = scipy.spatial.Delaunay(points)
        neighbor_vertices = tri.vertex_neighbor_vertices
        wall = self.Loc['plasma', 'poly'][0].poly.boundary
        boundary_vertices = np.array([i for i, polygon in
                                      enumerate(self.loc['plasma', 'poly'])
                                      if polygon.poly.intersects(wall)])
        centroids = np.array([np.mean(points[simplex], axis=0)
                              for simplex in tri.simplices])
        inside = PointLoop(centroids).update(np.array(wall.xy).T)
        triangles = tri.simplices[inside]
        stencil, stencil_index = self.loop_neighbour_vertices(
            points, neighbor_vertices, boundary_vertices)
        self.data.coords['x'] = points[:, 0]
        self.data.coords['z'] = points[:, 1]
        self.data.coords['stencil_index'] = stencil_index
        self.data['triangles'] = ('tri_index', 'tri_vertex'), triangles
        self.data['stencil'] = ('stencil_index', 'stencil_vertex'), stencil

    def plot(self, **kwargs):
        """Plot poloidal flux contours."""
        super().plot(axes=kwargs.get('axes', None))
        kwargs = self.contour_kwargs(**kwargs)
        if kwargs.get('plot_mesh', False):
            self.axes.triplot(self.data.x, self.data.z,
                              self.data.triangles, lw=0.5)
        self.axes.tricontour(self.data.x, self.data.z, self.data.triangles,
                             self.psi, **kwargs)


@dataclass
class BiotPlasmaVTK(BiotPlasmaGrid):
    """Extend BiotPlasmaGrid dataset with VTK methods."""

    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    mesh: pyvista.PolyData = field(init=False, repr=False)
    classnames: ClassVar[list[str]] = ['PlasmaGrid', 'PlasmaVTK']

    def __post_init__(self):
        """Load biot dataset."""
        super().__post_init__()
        self.load_data()
        assert self.data.attrs['classname'] in self.classnames
        self.build_mesh()

    def build_mesh(self):
        """Build vtk mesh."""
        points = np.c_[self.data.x, np.zeros(self.data.dims['x']), self.data.z]
        faces = np.c_[np.full(self.data.dims['tri_index'], 3),
                      self.data.triangles]
        self.mesh = pyvista.PolyData(points, faces=faces)

    def plot(self, **kwargs):
        """Plot vtk mesh."""
        self.mesh['psi'] = self.psi
        kwargs = dict(color='purple', line_width=2,
                      render_lines_as_tubes=True) | kwargs
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.mesh)
        plotter.add_mesh(self.mesh.contour(), **kwargs)
        plotter.show()
