"""Manage contouring algorithums solved on a rectngular plasma grid."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nova.biot.biotgrid import BiotGrid
from nova.biot.contour import Contour
from nova.frame.baseplot import Plot

# pylint: disable=too-many-ancestors


@dataclass
class LCFS(Plot):
    """Calculate plasma shape parameters from the last closed flux surface."""

    points: np.ndarray

    def __call__(self, attrs: list[str]):
        """Return attribute shape vector."""
        return np.array([getattr(self, attr) for attr in attrs])

    @cached_property
    def radius(self):
        """Return surface radius."""
        return self.points[:, 0]

    @cached_property
    def height(self):
        """Return surface height."""
        return self.points[:, 1]

    @cached_property
    def r_max(self):
        """Return maximum radius, Rmax."""
        return self.radius.max()

    @cached_property
    def r_min(self):
        """Return minimum radius, Rmin."""
        return self.radius.min()

    @cached_property
    def geometric_radius(self):
        """Return geometric radius, Rgeo."""
        return (self.r_max + self.r_min) / 2

    @cached_property
    def minor_radius(self):
        """Return minor radius, a."""
        return (self.r_max - self.r_min) / 2

    @cached_property
    def z_max(self):
        """Return maximum height, Zmax."""
        return self.height.max()

    @cached_property
    def z_min(self):
        """Return minimum height, Zmin."""
        return self.height.min()

    @cached_property
    def inverse_aspect_ratio(self):
        """Return inverse aspect ratio, epsilon."""
        return self.minor_radius / self.geometric_radius

    @cached_property
    def elongation(self):
        """Return elongation, kappa."""
        return (self.z_max - self.z_min) / (2*self.minor_radius)

    @cached_property
    def r_zmax(self):
        """Return radius at maximum height, Rzmax."""
        return self.radius[np.argmax(self.height)]

    @cached_property
    def r_zmin(self):
        """Return radius at minimum height, Rzmin."""
        return self.radius[np.argmin(self.height)]

    @cached_property
    def triangularity(self):
        """Return triangularity, del."""
        r_zmean = (self.r_zmax + self.r_zmin) / 2
        return (self.geometric_radius - r_zmean) / self.minor_radius

    @cached_property
    def upper_triangularity(self):
        """Return upper triangularity, del_u."""
        return (self.geometric_radius - self.r_zmax) / self.minor_radius

    @cached_property
    def lower_triangularity(self):
        """Return lower triangularity, del_l."""
        return (self.geometric_radius - self.r_zmin) / self.minor_radius

    def plot(self):
        """Plot last closed flux surface and key geometrical points."""
        self.get_axes('2d')
        self.axes.plot(self.radius, self.height, color='k', alpha=0.25)
        self.axes.plot(self.r_max, self.height[np.argmax(self.radius)], 'o',
                       label='Rmax')
        self.axes.plot(self.r_min, self.height[np.argmin(self.radius)], 'o',
                       label='Rmin')
        self.axes.plot(self.geometric_radius, (self.z_max + self.z_min) / 2,
                       'o', label='Rgeo')
        self.axes.plot(self.r_zmax, self.z_max, 'o', label='Zmax')
        self.axes.plot(self.r_zmin, self.z_min, 'o', label='Zmin')
        self.axes.legend(loc='center')


@dataclass
class Flux(BiotGrid):
    """Extend BiotGrid class with flux contouring algorithums."""

    nplasma: float = 1000
    levels: int | np.ndarray = 50
    contour: Contour = field(init=False, repr=False)

    def solve(self, *args, limit=0, index='plasma'):
        """Solve rectangular grid fit to first wall contour."""
        try:
            ngrid = args[0]
        except IndexError:
            ngrid = self.nplasma
        super().solve(ngrid, limit=limit, index=index)

    def load_operators(self):
        """Extend BiotGrid.load_operators to initalize contour instance."""
        super().load_operators()
        self.contour = Contour(self.data.x2d, self.data.z2d, self.psi_,
                               levels=self.levels)

    def check_contour(self):
        """Check contour flux operators."""
        self.check_plasma('Psi')
        self.check_source('psi')

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == 'contour':
            self.check_contour()
        return super().__getattribute__(attr)

    def lcfs(self, psi):
        """Return last closed flux surface."""
        return LCFS(self.contour.closedlevelset(psi))
