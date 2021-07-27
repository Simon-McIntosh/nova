"""Build interaction matrix toroidal loops."""
from dataclasses import dataclass


from nova.electromagnetic.biotfilament import Biot
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.polyplot import Axes


@dataclass
class BiotInductance(Axes, BiotSolve):
    """Compute self interaction."""

    def solve(self, index=slice(None)):
        """Solve Biot interaction across grid."""
        self.data = Biot(self.subframe, self.subframe.loc[:, index],
                         reduce=[True, True], columns=['Psi']).data

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='') | kwargs
        self.axes.plot(self.subframe['x'], self.subframe['z'], **kwargs)
