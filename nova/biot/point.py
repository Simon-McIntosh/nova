"""Build interaction matrix for a set of poloidal points."""
from dataclasses import dataclass

import numpy as np

from nova.biot.biotframe import Target
from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.graphics.plot import Plot


@dataclass
class Point(Plot, Operate):
    """Compute interaction for a series of discrete points."""

    def solve(self, points):
        """Solve Biot interaction at points."""
        points = np.array(points)
        points.shape = (-1, 2)  # shape(n, 2)
        target = Target(dict(x=[point[0] for point in points],
                             z=[point[1] for point in points]), label='Point')
        self.data = Solve(self.subframe, target, reduce=[True, False],
                          attrs=self.attrs, name=self.name).data
        # insert coordinate data
        self.data.coords['x'] = target['x']
        self.data.coords['z'] = target['z']
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='', color='C1') | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)
