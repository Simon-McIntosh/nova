"""Build interaction matrix for a set of poloidal points."""
from dataclasses import dataclass

# import nlopt
import numpy as np

from nova.biot.biotframe import BiotTarget
from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.frame.polyplot import Axes


@dataclass
class BiotPoint(Axes, BiotOperate):
    """Compute interaction for a series of discrete points."""

    def solve(self, points):
        """Solve Biot interaction at points."""
        points = np.array(points)
        points.shape = (-1, 2)  # shape(n, 2)
        target = BiotTarget(dict(x=[point[0] for point in points],
                                 z=[point[1] for point in points]),
                            label='Point')
        self.data = BiotSolve(self.subframe, target, reduce=[True, False],
                              attrs=['Psi', 'Br', 'Bz'],
                              name=self.name).data
        # insert coordinate data
        self.data.coords['x'] = target['x']
        self.data.coords['z'] = target['z']
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='', color='C1') | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)
