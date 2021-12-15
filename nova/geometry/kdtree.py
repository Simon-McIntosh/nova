"""Build fast-access KDtree lookups to reduce point-in-polygon search space."""
from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt
import scipy.spatial

from nova.electromagnetic.coilset import CoilSet
from nova.geometry.polygon import Polygon
from nova.utilities.pyplot import plt

from nova.geometry.inpoly import polymultipoint


@dataclass
class Loop:
    """Construct bounding loop."""

    points: npt.ArrayLike

    def __post_init__(self):
        """Calculate boundary."""
        self.bounds = np.array([self.points[:, 0].min(),
                                self.points[:, 0].max(),
                                self.points[:, 1].min(),
                                self.points[:, 1].max()])
        self.delta = self.bounds[1::2] - self.bounds[::2]
        self.center = self.bounds[::2] + self.delta/2

    def line(self, index: int):
        """Return 1D bounding box center and radius."""
        return self.center[index], self.delta[index]/2

    def circle(self):
        """Return center and radius of interior circle."""
        radius = np.min(np.linalg.norm(self.points-self.center, axis=1))
        return self.center, radius


@dataclass
class KDTree:
    """Implement fast point-in-polygon lookup using KDtrees."""

    points: npt.ArrayLike
    eps: float
    leafsize: int = 2
    tree: dict[str, scipy.spatial.KDTree] = field(init=False,
                                                  default_factory=dict)

    def __post_init__(self):
        self.mask = np.full(len(self.points), False)
        self.build_trees()

    def build_trees(self, leafsize=2):
        """Construct 1D and 2D KDtrees."""
        self.tree['x'] = scipy.spatial.KDTree(self.points[:, :1], leafsize)
        self.tree['z'] = scipy.spatial.KDTree(self.points[:, -1:], leafsize)
        self.tree['xz'] = scipy.spatial.KDTree(self.points, leafsize)

    def query_tree(self, name, center, radius, invert=False):
        """Return boolean mask from KDTree."""
        index = self.tree[name].query_ball_point(center, radius)
        return self._mask(index, invert)

    def _mask(self, index, invert=False):
        """Return boolean mask with mask[index]=True."""
        mask = self.mask.copy()
        mask[index] = True
        if invert:
            return ~mask
        return mask

    def query(self, points):
        """ """
        #loop = Loop(points)
        #return self.query_tree('xz', *loop.circle(), True)

        #radius = loop.circle()[1]
        #return np.linalg.norm(self.points-loop.center, axis=1) > radius

        return polymultipoint(self.points, points)


if __name__ == '__main__':

    coilset = CoilSet(dcoil=-35, dplasma=-250)
    coilset.plasma.insert(dict(e=(1, 2, 1.6, 0.8)))

    tree = KDTree(coilset.loc['plasma', ['x', 'z']].to_numpy(),
                  coilset.loc['plasma', 'dl'][0])
    separatrix = Polygon(dict(h=(1, 2, 0.4))).boundary
    separatrix = Polygon(dict(c=[1, 1.5, 0.8])).boundary

    mask = tree.query(separatrix)
    plt.plot(coilset.loc['plasma', 'x'][mask],
             coilset.loc['plasma', 'z'][mask], 'o')

    coilset.plot()
    plt.plot(*separatrix.T)
