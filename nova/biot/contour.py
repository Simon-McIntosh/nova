"""Extract flux function coefficnets from poloidal flux contours."""
from __future__ import annotations
import bisect
from dataclasses import dataclass, field, fields, InitVar
from typing import Callable

import contourpy
import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline, interp1d

from nova.plot.biotplot import LinePlot


@dataclass
class Surface(LinePlot):
    """Provide storage and transforms for single contour surface."""

    points: np.ndarray = field(repr=False)
    code: InitVar[np.ndarray] = field(repr=False)
    psi: float
    closed: bool = field(init=False)
    variable: np.ndarray = field(init=False, repr=False)
    coefficients: np.ndarray = field(init=False, repr=False)
    residual: np.ndarray = field(init=False, repr=False)

    def __post_init__(self, code):
        """Store contour surface."""
        super().__post_init__()
        self.closed = code[-1] == 79
        if len(self.points) <= 5:
            self.closed = False
        if not self.closed:
            self.linestyle = 'dashed'

    @property
    def radius(self):
        """Return contour surface radius."""
        return self.points[:, 0]

    @property
    def height(self):
        """Return contour surface height."""
        return self.points[:, 1]

    def fit(self, fun: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """Fit flux function coefficents to variable."""
        self.variable = fun(self.radius, self.height)
        self.coefficients, self.residual = np.linalg.lstsq(
            np.c_[self.radius, 1 / (mu_0 * self.radius)], self.variable)[:2]
        self.coefficients /= -2*np.pi
        if len(self.radius) < 5:
            self.coefficients = np.zeros(2)

    def plot(self, **kwargs):
        """Plot contour surface."""
        self.get_axes('2d')
        self.axes.plot(self.radius, self.height,
                       **self.plot_kwargs(**kwargs))


@dataclass
class ContourLoc:
    """Provide dict like access to contour data."""

    points: np.ndarray = field(init=False, repr=False)
    closed: np.ndarray = field(init=False, repr=False)
    psi: np.ndarray = field(init=False, repr=False)
    variable: np.ndarray = field(init=False, repr=False)
    coefficients: np.ndarray = field(init=False, repr=False)
    residual: np.ndarray = field(init=False, repr=False)

    def __getitem__(self, attr: str):
        """Implement dict like acces to contour attributes."""
        return getattr(self, attr)

    def update(self, nest: list[Surface]):
        """Update contour attributes."""
        for attr in (field.name for field in fields(ContourLoc)):
            setattr(self, attr,
                    np.array([getattr(surface, attr) for surface in nest]))


@dataclass
class Contour(LinePlot):
    """Contour 2d poloidal flux map and extract flux function coeffiecents."""

    x2d: np.ndarray
    z2d: np.ndarray
    psi2d: np.ndarray
    levels: int | np.ndarray = 10
    nest: list[Surface] = field(init=False, repr=False)
    loc: ContourLoc = field(init=False, default_factory=ContourLoc)

    def __post_init__(self):
        """Initialize contour generator."""
        super().__post_init__()
        self.generator = contourpy.contour_generator(
            self.x2d, self.z2d, self.psi2d,
            line_type='SeparateCode', quad_as_tri=True)

    @property
    def psi(self):
        """Return poloidal flux contor levels."""
        match self.levels:
            case int():
                return np.linspace(self.psi2d.min(),
                                   self.psi2d.max(),
                                   self.levels)
            case np.ndarray():
                return self.levels
            case _:
                raise TypeError('levels has incorect type '
                                f'{type(self.levels)}')

    def update_loc(self):
        """Update loc indexer."""
        self.loc.update(self.nest)

    def levelset(self, psi):
        """Return level-set from contour nest."""
        nest = []
        for points, code in zip(*self.generator.lines(psi)):
            nest.append(Surface(points, code, psi))
        return nest

    def closedlevelset(self, psi):
        """Return first closed level-set contour nest."""
        for points, code in zip(*self.generator.lines(psi)):
            if (surface := Surface(points, code, psi)).closed:
                return surface
        return surface

    def plot_levelset(self, psi, closed=True, **kwargs):
        """Plot contours for single levelset."""
        for surface in self.levelset(psi):
            if closed and not surface.closed:
                continue
            surface.plot(**kwargs)

    def generate(self, fun=None):
        """Generate contours."""
        self.nest = []
        for psi in self.psi:
            for points, code in zip(*self.generator.lines(psi)):
                surface = Surface(points, code, psi)
                if fun is not None:
                    surface.fit(fun)
                self.nest.append(surface)
        self.update_loc()

    def plot(self, **kwargs):
        """Plot contours."""
        for surface in self.nest:
            surface.plot(**self.plot_kwargs(**kwargs))

    def plot_fit(self, psi: float, norm: Callable | None = None, axes=None):
        """Plot least squares fit for single closed contour."""
        contour_psi = self.loc['psi'][self.loc['closed']]
        if norm is not None:
            contour_psi = norm(contour_psi)
        if contour_psi[0] < contour_psi[-1]:
            index = bisect.bisect_left(contour_psi, psi)
        else:
            index = bisect.bisect_right(-contour_psi, -psi)
        surface = np.array(self.nest)[self.loc['closed']][index]
        self.set_axes('1d', axes=axes)
        self.axes.plot(surface.points[:, 0], surface.variable, 'C0.', ms=5,
                       label=rf'contour: $\psi^\prime$={psi}')
        radius = np.linspace(np.min(surface.points[:, 0]),
                             np.max(surface.points[:, 0]))

        fit = -2*np.pi * np.c_[radius, 1 / (mu_0 * radius)] @ surface.coefficients
        fit_label = rf'fit: $p^\prime$={surface.coefficients[0]:1.2f} '
        fit_label += rf'$ff^\prime$={surface.coefficients[1]:1.2f}'
        self.axes.plot(radius, fit, 'C1', label=fit_label)
        self.axes.legend()
        self.axes.set_xlabel('radius')
        self.axes.set_ylabel(r'$J_p$')
        return np.arange(len(self.loc['psi']))[self.loc['closed']][index]

    def plot_1d(self, norm: Callable | None = None, index: int = 0,
                axes=None, **kwargs):
        """Plot flux function fit."""
        self.set_axes('1d', axes=axes)
        psi = self.loc['psi']
        if norm is not None:
            psi = norm(psi)  # COCOS
        self.axes.plot(psi[self.loc['closed']],
                       self.loc['coefficients'][self.loc['closed'], index],
                       **kwargs)


if __name__ == '__main__':

    from nova.frame.coilset import CoilSet
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(3, 0, 0.4, 0.2, Ic=1)
    coilset.grid.solve(250, 0.5)

    coilset.plot()
    levels = coilset.grid.plot()

    contour = Contour(coilset.grid.data.x2d, coilset.grid.data.z2d,
                      coilset.grid.psi_, levels=levels)

    Jp_fun = RectBivariateSpline(coilset.grid.data.x, coilset.grid.data.z,
                                 coilset.grid.data.x2d).ev

    contour.generate(Jp_fun)

    contour.plot_1d()
