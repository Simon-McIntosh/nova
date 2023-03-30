"""Manage kd-tree selection methods."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.spatial

from nova.frame.baseplot import Plot


@dataclass
class KDTree(Plot):
    """Implement fast kd-tree search."""

    points: np.ndarray
    factor: float = 0.1
    number: int = field(init=False, default=0)

    def __post_init__(self):
        """Update point number."""
        self.number = len(self.points)
        super().__post_init__()

    @cached_property
    def kd_tree(self):
        """Return 2d plasma filament selection tree."""
        return scipy.spatial.KDTree(self.points)

    @cached_property
    def upper_bound(self):
        """Return 2d plasma filament selection tree distance upper bound."""
        return self.factor * self.delta

    @cached_property
    def delta(self):
        """Return mean adjacent node spacing."""
        return np.mean(self.kd_tree.query(self.points, k=2)[0][..., 1])

    def query(self, other: np.ndarray) -> np.ndarray:
        """Return distance plasma filament selection arrays."""
        distance, tree = self.kd_tree.query(
            other, distance_upper_bound=self.upper_bound)
        unique, unique_indices = np.unique(tree, return_index=True)
        missing = unique == self.number
        index = unique_indices[~missing]
        return distance[index], tree[index]

    def plot(self, other: np.ndarray):
        """Plot kd selection query."""
        self.get_axes('2d')
        index = self.query(other)[1]
        self.axes.plot(*other.T, '-.', color='gray')
        self.axes.plot(*self.points[index, :].T, 'k.')
