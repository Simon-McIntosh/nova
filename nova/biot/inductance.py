"""Build interaction matrix toroidal loops."""
from dataclasses import dataclass

from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.frame.baseplot import Plot
from nova.frame.polygrid import PolyTarget


@dataclass
class Inductance(Plot, Operate):
    """Compute self interaction."""

    def solve(self, number=None, index=slice(None)):
        """Solve Biot interaction across subframe."""
        with self.solve_biot(number) as number:
            if number is not None:
                self.target = PolyTarget(self.Loc[:, index], -number).target
                self.data = Solve(self.subframe, self.target,
                                  turns=[True, True], reduce=[True, True],
                                  attrs=self.attrs, name=self.name).data

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='') | kwargs
        self.axes.plot(self.subframe['x'], self.subframe['z'], **kwargs)
