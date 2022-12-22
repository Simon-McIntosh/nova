"""Extract flux function coefficnets from poloidal flux contours."""
from __future__ import annotations
from dataclasses import dataclass, field, fields, InitVar
from typing import TYPE_CHECKING

import contourpy
import numpy as np

from nova.plot.biotplot import BiotPlot, LinePlot

if TYPE_CHECKING:
    from nova.biot.biotgrid import BiotGrid


@dataclass
class Line(LinePlot):
    """Provide storage and transforms for single contour line."""

    points: np.ndarray
    code: InitVar[np.ndarray]
    closed: bool = field(init=False)

    def __post_init__(self, code):
        """Store contour line."""
        super().__post_init__()
        self.closed = code[-1] == 79
        if not self.closed:
            self.linestyle = 'dashed'

    @property
    def radius(self):
        """Return contour line radius."""
        return self.points[:, 0]

    @property
    def height(self):
        """Return contour line height."""
        return self.points[:, 1]

    def plot(self, **kwargs):
        """Plot contour line."""
        self.axes.plot(self.radius, self.height,
                       **self.plot_kwargs(**kwargs))


@dataclass
class ContourLoc:
    """Provide dict like access to contour data."""

    radius: np.ndarray = field(init=False, repr=False)
    height: np.ndarray = field(init=False, repr=False)
    closed: np.ndarray = field(init=False, repr=False)

    def __getitem__(self, attr: str):
        """Implement dict like acces to contour attributes."""
        return getattr(self, attr)

    def update(self, lines: list[Line]):
        """Update contour attributes."""
        for attr in (field.name for field in fields(ContourLoc)):
            setattr(self, attr,
                    np.array([getattr(line, attr) for line in lines]))


@dataclass
class Contour(LinePlot):
    """Contour 2d poloidal flux map and extract flux function coeffiecents."""

    grid: BiotGrid
    levels: int | np.ndarray = 10
    lines: list[Line] = field(init=False, repr=False)
    loc: ContourLoc = field(init=False, default_factory=ContourLoc)

    def __post_init__(self):
        """Initialize contour generator."""
        super().__post_init__()
        self.generator = contourpy.contour_generator(
            self.grid.data.x2d, self.grid.data.z2d, self.grid.psi_,
            line_type='SeparateCode')

    @property
    def psi(self):
        """Return poloidal flux contor levels."""
        match self.levels:
            case int():
                return np.linspace(self.grid.psi.min(), self.grid.psi.max(),
                                   self.levels)
            case np.ndarray():
                return self.levels
            case _:
                raise TypeError('levels has incorect type '
                                f'{type(self.levels)}')

    def update(self):
        """Update loc indexer."""
        self.loc.update(self.lines)

    def generate(self):
        """Extract contours."""
        self.lines = []
        for psi in self.psi:
            for points, code in zip(*self.generator.lines(psi)):
                self.lines.append(Line(points, code))
        self.update()

    def plot(self, **kwargs):
        """Plot contours."""
        for line, closed in zip(self.lines, self.loc['closed']):
            if closed:
                line.plot(color='C0')


if __name__ == '__main__':

    from nova.frame.coilset import CoilSet
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(3, 0, 0.4, 0.2, Ic=1)
    coilset.grid.solve(250, 0.5)

    coilset.plot()
    levels = coilset.grid.plot()

    contour = Contour(coilset.grid, levels=levels)
    contour.generate()

    contour.plot()

    Jp = operate._rbs('j_tor2d')(radius, height)
    alpha, beta = np.linalg.lstsq(np.c_[radius, radius**-1], Jp)[0]

    '''
    coords, code = cgen.lines(73)

    axes_2d = operate.set_axes(operate.axes, '2d')
    axes_1d = operate.set_axes(None, '1d')

    for index in range(len(coords)):
        radius = coords[index][:, 0]
        height = coords[index][:, 1]

        axes_2d.plot(radius, height, 'C3')

        Jp = operate._rbs('j_tor2d')(radius, height)

        #Jp -= 2e3*height

        #operate.set_axes(None, '1d')
        axes_1d.plot(radius, Jp, 'C3')

        alpha, beta = np.linalg.lstsq(np.c_[radius, radius**-1], Jp)[0]

        axes_1d.plot(radius := np.linspace(radius.min(), radius.max()),
                     alpha*radius + beta/radius, 'C0--')
    '''
