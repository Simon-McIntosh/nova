"""Generate of an artifical an separatrix from shape parameters."""
from dataclasses import dataclass, field, InitVar

from nova.biot.flux import LCFS

import numpy as np


@dataclass
class Separatrix:
    """Generate target separatrix from plasma shape parameters."""

    geometric_radius: float = 0
    geometric_height: float = 0
    minor_radius: float = 1
    elongation: float | tuple[float, float] = 0
    triangularity: float | tuple[float, float] = 0
    number: int = 250
    lcfs: LCFS = field(init=False, repr=False)

    def __post_init__(self):
        """Build seperatrix."""
        self.build()

    def __call__(self, attrs):
        """Return lcfs attribute list."""
        return self.lcfs(attrs)

    def __getitem__(self, attr):
        """Return lcfs attributes."""
        return getattr(self.lcfs, attr)

    def build(self):
        """Build canditate surface."""
        theta = np.linspace(0, 2*np.pi, self.number)
        del_hat = np.arcsin(self.triangularity)
        points = self.minor_radius * np.c_[
             np.cos(theta + del_hat * np.sin(theta)),
             self.elongation * np.sin(theta)]
        points[:, 0] += self.geometric_radius
        points[:, 1] += self.geometric_height
        self.lcfs = LCFS(points)

    def plot(self):
        """Plot last closed flux surface."""
        self.lcfs.plot()


if __name__ == '__main__':

    separatrix = Separatrix(5, 2, 3, 1, 0.5)
    print(separatrix(['geometric_radius', 'minor_radius',
                      'elongation', 'triangularity']))
    separatrix.plot()
