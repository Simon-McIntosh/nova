"""Build interaction matrix toroidal loops."""

from dataclasses import dataclass, field


from nova.biot.biotframe import Target
from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.graphics.plot import Plot


@dataclass
class Loop(Plot, Operate):
    """Compute interaction through diagnostic flux_loops."""

    target: Target = field(init=False, repr=False, default_factory=Target)

    def insert(self, *args, dl=0, dt=0, **kwargs):
        """Insert flux loop into target."""
        self.target.insert(*args, dl=dl, dt=dt, **kwargs)

    def solve(self):
        """Solve Biot interaction at targets."""
        with self.solve_biot(len(self.target)) as number:
            if number == 0:
                return
            self.data = Solve(
                self.subframe,
                self.target,
                reduce=[True, True],
                turns=[True, True],
                attrs=["Psi"],
                name=self.name,
            ).data
            # insert target data
            self.data.coords["x"] = self.target.x
            self.data.coords["z"] = self.target.z

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker="X", linestyle="", color="gray") | kwargs
        if len(self.data) == 0:
            self.axes.plot(self.target.x, self.target.z, **kwargs)
            return
        self.axes.plot(self.data.x, self.data.z, **kwargs)
