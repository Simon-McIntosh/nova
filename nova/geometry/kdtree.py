"""Manage kd-tree selection methods."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.spatial

from nova.graphics.plot import Plot


@dataclass
class KDTree(Plot):
    """Implement fast kd-tree search."""

    points: np.ndarray
    factor: float = 0.1
    number: int = field(init=False, default=0)

    def __post_init__(self):
        """Update point number."""
        self.number = len(self.points)

    @cached_property
    def kd_tree(self):
        """Return 2d selection tree."""
        return scipy.spatial.KDTree(self.points)

    @cached_property
    def upper_bound(self):
        """Return 2d plasma filament selection tree distance upper bound."""
        return self.factor * self.delta

    @cached_property
    def delta(self):
        """Return mean adjacent node spacing."""
        return np.mean(self.kd_tree.query(self.points, k=2)[0][..., 1])

    def query(self, other: np.ndarray):
        """Return distance plasma filament selection arrays."""
        distance, tree = self.kd_tree.query(
            other, distance_upper_bound=self.upper_bound)
        unique, unique_indices = np.unique(tree, return_index=True)
        missing = unique == self.number
        index = unique_indices[~missing]
        try:
            return distance[index], tree[index]
        except TypeError:
            pass
        if index[0] == 0:
            return distance, tree
        return np.array([]), np.array([])

    def plot(self, other: np.ndarray):
        """Plot kd selection query."""
        self.get_axes('2d')
        index = self.query(other)[1]
        self.axes.plot(*other.T, '-.', color='gray')
        self.axes.plot(*self.points[index, :].T, 'k.')


@dataclass
class Proximate:
    """Implement fast nearest-neighbour selections via kd-tree queries."""

    kd_factor: float = 0.1
    kd_tree: KDTree = field(init=False, repr=False)

    @property
    def kd_points(self):
        """Generate kd-tree and return point cloud."""
        return self.kdtree.points

    @kd_points.setter
    def kd_points(self, points):
        self._update_kd_tree(points)

    def _update_kd_tree(self, points: np.ndarray):
        """Update kd selection tree."""
        self.kd_tree = KDTree(points, self.kd_factor)

    def kd_query(self, other: np.ndarray):
        """Return point index from kdtree."""
        return self.kd_tree.query(other)[1]
