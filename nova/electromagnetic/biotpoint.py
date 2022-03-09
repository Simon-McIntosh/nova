"""Build interaction matrix for a set of poloidal points."""
from dataclasses import dataclass, field

import nlopt
import numpy as np

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotoperate import BiotOperate
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.biotsolve import BiotMatrix
from nova.electromagnetic.polyplot import Axes


@dataclass
class BiotPoint(Axes, BiotOperate):
    """Compute interaction for a series of discrete points."""

    def solve(self, points):
        """Solve Biot interaction at points."""
        points = np.array(points)
        points.shape = (-1, 2)  # shape(n, 2)
        target = BiotFrame(dict(x=[point[0] for point in points],
                                z=[point[1] for point in points]),
                           label='Point')
        self.data = BiotSolve(self.subframe, target, reduce=[True, False],
                              attrs=['Psi', 'Br', 'Bz']).data
        # insert coordinate data
        self.data.coords['x'] = target['x'].values
        self.data.coords['z'] = target['z'].values
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='') | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)


@dataclass
class PointInverse(FrameSetLoc):
    """Flux matching inverse solution for free coil currents."""

    data: BiotMatrix = field(repr=False, default=None)
    gamma: float = 0

    def __post_init__(self):
        self.set_foreground()

    def set_foreground(self):
        '[G][Ic] = [T]'
        self.G = self.data['Psi'][:, self.sloc['free']].values

    def set_background(self):
        'contribution from passive coils'
        self.BG = self.data['Psi'][:, self.sloc['fix']].values @ \
            self.sloc['fix', 'Ic']

    def update_target(self, Psi):
        self.T = Psi - self.BG

    def update(self, Psi):
        self.set_background()
        self.update_target(Psi)

    @property
    def err(self):
        """Return error vector."""
        return self.G @ self.sloc['free', 'Ic'] - self.T

    @property
    def rss(self):
        'residual sum of squares with Tikhonov regularization'
        return np.sum(self.err**2) + self.gamma * np.sum(self.sloc['Ic']**2)

    def frss(self, Ic, grad):
        self.Ic = Ic  # update current vector
        if grad.size > 0:
            jac = 2 * self.G.T @ self.G @ self.sloc['Ic']
            jac -= 2 * self.G.T @ self.T
            jac += self.gamma * 2 * self.sloc['Ic']  # Tikhonov regularization
            grad[:] = jac  # set gradient in-place
        return self.rss

    def solve_lstsq(self, Psi):
        'linear least squares solution'
        self.update(Psi)
        self.sloc['free', 'Ic'] = \
            np.linalg.lstsq(self.G, self.T, rcond=None)[0]

    def solve(self, Psi):  # solve for constrained current vector
        self.update(Psi)
        opt = nlopt.opt(nlopt.LD_MMA, np.sum(self.sloc['free']))
        opt.set_min_objective(self.frss)
        opt.set_ftol_rel(1e-6)
        self.sloc['free', 'Ic'] = opt.optimize(self.sloc['free', 'Ic'])
