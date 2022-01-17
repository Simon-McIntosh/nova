"""Generate grid and solution methods for hexagonal plasma filaments."""
from dataclasses import dataclass, field
from typing import Union

import numba
import numpy as np
from numpy import typing as npt
import scipy.spatial
import xarray

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotoperate import BiotOperate
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.fieldnull import FieldNull


@dataclass
class PlasmaGrid(FieldNull, BiotOperate):
    """Compute interaction across hexagonal grid."""

    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    levels: Union[int, list[float]] = 31
    x_coordinate: npt.ArrayLike = field(repr=False, default=None)
    z_coordinate: npt.ArrayLike = field(repr=False, default=None)

    def __post_init__(self):
        """Initialize fieldnull version."""
        super().__post_init__()
        self.version['fieldnull'] = id(None)

    @staticmethod
    @numba.njit
    def loop_neighbour_vertices(points, neighbor_vertices):
        """Calculate 6-point ordered loop vertex indices."""
        point_number = len(points)
        neighbours = np.full((point_number, 6), -1)
        for i in range(len(points)):
            center_point = points[i, :]
            slice_index = slice(neighbor_vertices[0][i],
                                neighbor_vertices[0][i+1])
            neighbour_index = neighbor_vertices[1][slice_index]
            if len(neighbour_index) != 6:
                continue
            delta = points[neighbour_index] - center_point
            angle = np.arctan2(delta[:, 1], delta[:, 0])
            neighbours[i] = neighbour_index[np.argsort(angle)[::-1]]
        index = neighbours[:, 0] != -1
        return np.append(np.arange(point_number)[index].reshape(-1, 1),
                         neighbours[index], axis=1)

    def tessellate(self):
        """Tesselate hexagonal mesh, compute 6-point neighbour loops."""
        points = self.subframe.loc['plasma', ['x', 'z']].to_numpy()
        tri = scipy.spatial.Delaunay(points)
        neighbor_vertices = tri.vertex_neighbor_vertices
        neighbours = self.loop_neighbour_vertices(points, neighbor_vertices)
        self.data.coords['x'] = points[:, 0]
        self.data.coords['z'] = points[:, 1]
        self.data['triangles'] = ('tri_index', 'tri_vertex'), tri.simplices
        self.data['stencil'] = ('stencil_index', 'stencil_vertex'), neighbours

    def solve(self,):
        """Solve Biot interaction across plasma grid."""
        target = BiotFrame(self.subframe.loc['plasma', ['x', 'z']].to_dict())
        self.data = BiotSolve(self.subframe, target, reduce=[True, False],
                              attrs=self.attrs).data
        self.tessellate()
        self.link_fieldnull()
        super().solve()

    def link_array(self):
        """Extend biot data link_array to link field null instance."""
        super().link_array()
        self.link_fieldnull()

    def link_fieldnull(self):
        """Link to field null instance."""
        self.x_coordinate = self.data.x.data
        self.z_coordinate = self.data.z.data
        self.stencil = self.data.stencil.data

    def plot(self, axes=None, **kwargs):
        """Plot poloidal flux contours."""
        self.axes = axes
        kwargs = dict(colors='lightgray', linewidths=1.5, alpha=0.9,
                      linestyles='solid', levels=self.levels) | kwargs
        '''

        psi = asnumpy(self.psi.reshape(*self.shape).T)
        QuadContourSet = self.axes.contour(self.data.x, self.data.z, psi,
                                           **kwargs)
        if isinstance(kwargs['levels'], int):
            self.levels = QuadContourSet.levels
        '''
        #self.axes.triplot(self.x_coordinate, self.z_coordinate,
        #                  self.data['triangles'].data, lw=0.5)
        self.axes.tricontour(self.x_coordinate, self.z_coordinate,
                             self.data['triangles'].data, self.psi,
                             **kwargs)
        super().plot()

        '''
        neighbours = loop_neighbour_vertices(points, neighbor_vertices)

        p = 20
        n = tri.vertex_neighbor_vertices
        npp = n[1][n[0][p]:n[0][p+1]]

        plt.plot(*points[p, :].T, 'C3o')
        plt.plot(*points[npp, :].T, 'C0o')

        angle = np.arctan2(*(points[npp, :] - points[p, :]).T)
        npp = npp[np.argsort(angle)[::-1]]

        npp = neighbours[20]

        for i, _np in enumerate(npp):
            plt.text(*points[_np, :].T, i)

            plt.axis('equal')


        '''
