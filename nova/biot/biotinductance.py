"""Build interaction matrix toroidal loops."""
from dataclasses import dataclass

from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.frame.baseplot import Plot


@dataclass
class BiotInductance(Plot, BiotOperate):
    """Compute self interaction."""

    def solve(self, index=slice(None)):
        """Solve Biot interaction across subframe."""
        self.data = BiotSolve(self.subframe, self.subframe.loc[:, index],
                              turns=[True, True], reduce=[True, True],
                              attrs=['Psi'], name=self.name).data
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='') | kwargs
        self.axes.plot(self.subframe['x'], self.subframe['z'], **kwargs)
