"""Build interaction matrix toroidal loops."""

from dataclasses import dataclass

from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.graphics.plot import Plot


@dataclass
class Loop(Plot, Operate):
    """Compute interaction across grid."""

    def solve(self, target):
        """Solve Biot interaction at targets."""
        self.data = Solve(
            self.subframe,
            target,
            reduce=[True, True],
            turns=[True, True],
            attrs=["Psi"],
            name=self.name,
        ).data
        # insert grid data
        self.data.coords["x"] = target.x.values
        self.data.coords["z"] = target.z.values
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker="o", linestyle="", color="C1") | kwargs
        self.axes.plot(self.data.coords["x"], self.data.coords["z"], **kwargs)
