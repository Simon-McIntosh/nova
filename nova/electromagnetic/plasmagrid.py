"""Generate grid and solution methods for hexagonal plasma filaments."""
from dataclasses import dataclass, field
from typing import Union

import numba
import numpy as np
import scipy.spatial

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotgrid import BiotBaseGrid
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.error import GridError
from nova.geometry.pointloop import PointLoop


@dataclass
class PlasmaGrid(BiotBaseGrid):
    """Compute interaction across hexagonal grid."""

    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    levels: Union[int, list[float]] = 21

    def __post_init__(self):
        """Initialize fieldnull version."""
        super().__post_init__()
        self.version['fieldnull'] = id(None)

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

    def solve(self,):
        """Solve Biot interaction across plasma grid."""
        if self.sloc['plasma'].sum() == 0:
            raise GridError('plasma')
        target = BiotFrame(self.subframe.loc['plasma', ['x', 'z']].to_dict())
        self.data = BiotSolve(self.subframe, target, reduce=[True, False],
                              attrs=self.attrs).data
        self.tessellate()
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot poloidal flux contours."""
        super().plot(axes)
        kwargs = self.contour_kwargs(**kwargs)
        if kwargs.get('plot_mesh', False):
            self.axes.triplot(self.data.x, self.data.z,
                              self.data.triangles, lw=0.5)
        self.axes.tricontour(self.data.x, self.data.z, self.data.triangles,
                             self.psi, **kwargs)
