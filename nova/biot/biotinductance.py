"""Build interaction matrix toroidal loops."""
from dataclasses import dataclass

from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.frame.baseplot import Plot
from nova.frame.polygrid import PolyTarget


@dataclass
class BiotInductance(Plot, BiotOperate):
    """Compute self interaction."""

    def solve(self, number=None, index=slice(None)):
        """Solve Biot interaction across subframe."""
        with self.solve_biot(number) as number:
            if number is not None:
                self.target = PolyTarget(self.Loc[:, index], -number).target
                self.data = BiotSolve(self.subframe, self.target,
                                      turns=[True, True], reduce=[True, True],
                                      attrs=self.attrs, name=self.name).data

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='') | kwargs
        self.axes.plot(self.subframe['x'], self.subframe['z'], **kwargs)
