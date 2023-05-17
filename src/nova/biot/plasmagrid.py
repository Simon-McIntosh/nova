"""Generate grid and solution methods for hexagonal plasma filaments."""
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module

import numpy as np
from shapely.geometry.linestring import LineString

from nova.biot.biotframe import Target
from nova.biot.error import PlasmaTopologyError
from nova.biot.grid import BaseGrid
from nova.biot.solve import Solve
from nova.frame.error import GridError
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop

from nova.frame.plasmaloc import PlasmaLoc


@dataclass
class PlasmaGrid(BaseGrid, PlasmaLoc):
    """Compute interaction across hexagonal grid."""

    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    levels: int | list[float] | np.ndarray = 21

    def __post_init__(self):
        """Initialize psi axis and psi x versions."""
        super().__post_init__()
        self.version['psi_axis'] = None
        self.version['psi_x'] = None

    def __getitem__(self, attr):
        """Implement dict-like access to plasmagrid attributes."""
        match attr:
            case 'x_point':
                return self.x_points[self._x_point_index()]
            case 'x_psi':
                return self.x_psi[self._x_point_index()]
            case 'o_point':
                return self.o_points[self._o_point_index()]
            case 'o_psi':
                return self.o_psi[self._o_point_index()]
        if hasattr(self, '__getitem__'):
            return super().__getitem__(attr)

    def _x_point_index(self):
        """Return x-point index for primary plasma separatrix."""
        match self.x_point_number:
            case 0:
                raise PlasmaTopologyError('no x-points within first wall')
        x_psi = self.x_psi.copy()
        if self.o_point_number > 0:
            x_psi - self.o_psi[0]
        return np.argmax(self.polarity*(x_psi))

    def _o_point_index(self):
        """Return plasma o-point index."""
        match self.o_point_number:
            case 1:
                return 0
            case 0:
                raise PlasmaTopologyError(
                    'no o-points found within first wall')
            case _:
                raise PlasmaTopologyError(
                    'multiple o-points found within first wall {self.data_o}')

    def solve(self):
        """Solve Biot interaction across plasma grid."""
        if self.sloc['plasma'].sum() == 0:
            raise GridError('plasma')
        target = Target(self.loc['plasma', ['x', 'z', 'poly']].to_dict())
        wall = self.Loc['plasma', 'poly'][0].poly.boundary
        self.data = Solve(self.subframe, target, reduce=[True, False],
                          attrs=self.attrs, name=self.name).data
        self.tessellate(target, wall)
        super().post_solve()

    @staticmethod
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

    def tessellate(self, target: Target, wall: LineString):
        """Tesselate hexagonal mesh, compute 6-point neighbour loops."""
        points = np.c_[target.x, target.z]
        tri = import_module('scipy.spatial').Delaunay(points)
        neighbor_vertices = tri.vertex_neighbor_vertices
        boundary_vertices = np.array([i for i, polygon in
                                      enumerate(target.poly)
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

    def psi_mask(self, psi):
        """Return plasma filament psi-mask."""
        if self.polarity > 0:
            return self.psi >= psi
        return self.psi < psi

    def x_mask(self, z_plasma: np.ndarray):
        """Return plasma filament x-mask."""
        mask = np.ones(len(z_plasma), dtype=bool)
        if self.x_point_number == 0 or self.o_point_number == 0:
            return mask
        o_point = self.o_points[0]
        for x_point in self.x_points:
            if x_point[1] < o_point[1]:
                mask &= z_plasma > x_point[1]
            else:
                mask &= z_plasma < x_point[1]
        return mask

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        if self.saloc['plasma'].sum() == 0:
            raise AttributeError('No plasma filaments found.')
        return PointLoop(self.loc['plasma', ['x', 'z']].to_numpy())

    def ionize_mask(self, index):
        """Return plasma filament selection mask."""
        match index:
            case int(psi) | float(psi):
                z_plasma = self.aloc['plasma', 'z']
                mask = self.psi_mask(psi)
                try:
                    return mask & self.x_mask(z_plasma)
                except IndexError:
                    return mask
            case [int(psi) | float(psi), float(z_min)]:
                return self.psi_mask(psi) & self.aloc['plasma', 'z'] > z_min
            case [int(psi) | float(psi), float(z_min), float(z_max)]:
                z_plasma = self.aloc['plasma', 'z']
                return self.psi_mask(psi) & z_plasma > z_min & z_plasma < z_max
            case _:
                try:
                    return self.pointloop.update(index)
                except Exception:  # numba.TypingError:
                    index = Polygon(index).boundary
                    return self.pointloop.update(index)

    def plot(self, **kwargs):
        """Plot poloidal flux contours."""
        super().plot(axes=kwargs.get('axes', None))
        kwargs = self.contour_kwargs(**kwargs)
        if kwargs.pop('plot_mesh', False):
            self.axes.triplot(self.data.x, self.data.z,
                              self.data.triangles, lw=0.5)
        self.axes.tricontour(self.data.x, self.data.z, self.data.triangles,
                             self.psi, **kwargs)
