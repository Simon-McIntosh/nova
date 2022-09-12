"""Build interaction matrix toroidal loops."""
from dataclasses import dataclass

from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.electromagnetic.polyplot import Axes


@dataclass
class BiotLoop(Axes, BiotOperate):
    """Compute interaction across grid."""

    def solve(self, target):
        """Solve Biot interaction at targets."""
        self.data = BiotSolve(self.subframe, target,
                              reduce=[True, True], turns=[True, True],
                              columns=['Psi']).data
        # insert grid data
        self.data.coords['x'] = target.x.values
        self.data.coords['z'] = target.z.values
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='', color='C1') | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)
