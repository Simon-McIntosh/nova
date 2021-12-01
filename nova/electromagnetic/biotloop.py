"""Build interaction matrix toroidal loops."""
from dataclasses import dataclass


from nova.electromagnetic.biotfilament import Biot
from nova.electromagnetic.biotdata import BiotData
from nova.electromagnetic.polyplot import Axes


@dataclass
class BiotLoop(Axes, BiotData):
    """Compute interaction across grid."""

    def _solve(self, target):
        """Solve Biot interaction."""
        self.data = Biot(self.subframe, target,
                         reduce=[True, True], turns=[True, True],
                         columns=['Psi']).data
        # insert grid data
        self.data.coords['x'] = target.x.values
        self.data.coords['z'] = target.z.values

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='') | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)
