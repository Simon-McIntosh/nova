"""Manage contouring algorithums solved on a rectngular plasma grid."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.spatial

from nova.biot.grid import Grid
from nova.biot.contour import Contour
from nova.biot.separatrix import PlasmaShape

# pylint: disable=too-many-ancestors


@dataclass
class LevelSet(Grid):
    """Extend Grid class with levelset contouring algorithums."""

    dlevelset: float = -0.25
    levels: int | np.ndarray = 50
    contour: Contour = field(init=False, repr=False)

    def solve(self, number=None, limit=0, index='plasma'):
        """Solve rectangular grid fit to first wall contour."""
        super().solve(number, limit=limit, index=index)

    def load_operators(self):
        """Extend Grid.load_operators to initalize contour instance."""
        super().load_operators()
        if self.number is not None:
            #self.number = self.data.dims['target']
            self.contour = Contour(self.data.x2d, self.data.z2d, self.psi_,
                                   levels=self.levels)

    def check_contour(self):
        """Check contour flux operators."""
        self.check_plasma('Psi')
        self.check_source('psi')

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == 'contour':
            self.check_contour()
        return super().__getattribute__(attr)

    def lcfs(self, psi):
        """Return last closed flux surface."""
        psi_levelset = self.contour.closedlevelset(psi)
        return PlasmaShape(psi_levelset.points)

    @cached_property
    def points(self):
        """Return flattened point array."""
        return np.c_[self.data.x2d.data.flatten(),
                     self.data.z2d.data.flatten()]

    @cached_property
    def kd_tree(self):
        """Return 2d plasma filament selection tree."""
        return scipy.spatial.KDTree(self.points)

    @cached_property
    def kd_upper_bound(self):
        """Return 2d plasma filament selection tree distance upper bound."""
        if self.dlevelset > 0:
            return self.dlevelset
        delta = np.mean(np.append(np.diff(self.data.x), np.diff(self.data.z)))
        return -self.dlevelset * delta

    def kd_query(self, points: np.ndarray) -> np.ndarray:
        """Return distance plasma filament selection arrays."""
        distance, tree = self.kd_tree.query(
            points, distance_upper_bound=self.kd_upper_bound)
        unique, unique_indices = np.unique(tree, return_index=True)
        missing = unique == self.number
        index = unique_indices[~missing]
        return distance[index], tree[index]

    def plot_query(self, points: np.ndarray):
        """Plot kq selection query."""
        self.get_axes('2d')
        index = self.kd_query(points)[1]
        self.axes.plot(*points.T, '-', color='gray')
        self.axes.plot(*self.points[index, :].T, 'k.')
