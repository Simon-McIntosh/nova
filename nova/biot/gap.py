"""Manage biot calculation plasma gap flux probes."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nova.biot.biotframe import Target
from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.frame.baseplot import Plot
from nova.geometry import select


@dataclass
class Gap(Plot, Operate):
    """Compute flux interaction across a series of 1d gap probes."""

    attrs: list[str] = field(default_factory=lambda: ['Psi'])
    ngap: int | float | None = 50
    mingap: int | float = 1e-3
    maxgap: int | float = 5
    gap_number: int = field(init=False, default=0)

    def __post_init__(self):
        """Update gap limits."""
        self.build()
        super().__post_init__()

    def build(self):
        """Extract gap limits from ngap."""
        match self.ngap:
            case int() if self.ngap > 0:
                self.gap_number = self.ngap
            case int() | float() if self.ngap < 0:
                self.gap_number = int(self.maxgap / -self.ngap) + 1
                self.mingap = 0
            case _:
                raise TypeError(f'invalid gap number {self.ngap}')

    @cached_property
    def gap(self):
        """Return gap node spacing."""
        if self.mingap == 0:
            return np.linspace(self.mingap, self.maxgap, self.gap_number)
        geomspace = np.geomspace(self.mingap, self.maxgap, self.gap_number-1)
        return np.append(0, geomspace)

    def solve(self, points, angle, label=None):
        """Solve linear gap flux probes."""
        if label is None:
            label = [f'Gap{i}' for i in range(len(points))]
        points = np.einsum('ik,j->ijk', points, np.ones(self.gap_number))
        probes = np.einsum('ik,j->ijk', np.c_[np.cos(angle), np.sin(angle)],
                           self.gap)
        target = Target()
        for gap, gap_label in zip(points + probes, label):
            target.insert(gap[:, 0], gap[:, 1], label=gap_label, delim='_')
        self.data = Solve(self.subframe, target, reduce=[True, False],
                          attrs=self.attrs, name=self.name).data
        self.data.coords['name'] = label
        self.data.coords['xo'] = 'name', points[:, 0, 0]
        self.data.coords['zo'] = 'name', points[:, 0, 1]
        self.data.coords['angle'] = 'name', angle
        self.data.coords['gap'] = self.gap
        self.data['x'] = 'target', target['x']
        self.data['z'] = 'target', target['z']
        self.data['x2d'] = ('name', 'gap'), points[..., 0]
        self.data['z2d'] = ('name', 'gap'), points[..., 1]
        self.data['index'] = ('name', 'gap'), \
            np.reshape(range(self.data.dims['target']), self.shape)
        super().post_solve()

    def load_operators(self):
        """Extend Grid.load_operators to initalize contour instance."""
        super().load_operators()
        if self.number is not None:
            self.gap_number = self.data.dims['gap']

    @cached_property
    def shape(self):
        """Return gap probe shape."""
        return self.data.dims['name'], self.data.dims['gap']

    def bisect(self, gap):
        """Return gap indicies."""
        return select.bisect_2d(self.gap, gap) + \
            self.gap_number * np.arange(self.data.dims['name'])

    def matrix(self, gap):
        """Return bisected Psi coupling matrix."""
        index = self.bisect(gap)
        return self.Psi[index, :]

    def plot(self, axes=None, **kwargs):
        """Plot wall-gap probes."""
        self.axes = axes
        self.axes.plot(self.data.x, self.data.z, '.', color='gray', ms=3)
