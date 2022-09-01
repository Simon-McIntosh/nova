"""Biot-Savart calculation for complete circular cylinders."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import dask.array as da
import numpy as np

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotmatrix import BiotMatrix


@dataclass
class BiotPolygon(BiotMatrix):
    """
    Extend Biotmatrix base class.

    Compute interaction for complete toroidal coils with polygonal sections.

    """

    edge: BiotFrame = field(repr=False, init=False)
    reduction_matrix: da.Array = field(repr=False, init=False)

    name: ClassVar[str] = 'polygon'  # element name
    attrs: ClassVar[dict[str, str]] = dict(area='area')
    metadata: ClassVar[dict[str, list]] = dict(
        required=['ref', 'r1', 'z1', 'r2', 'z2'], additional=[],
        available=[], array=['ref'], base=[])

    def __post_init__(self):
        """Initialize biotset and edgeframe."""
        super().__post_init__()
        self.build()

    def __getattr__(self, attr):
        """Return data attributes."""
        return self.data[attr]

    @contextmanager
    def target_edge(self):
        """Manage target to edge tile number."""
        self.target.set_source(len(self.edge))
        yield
        self.target.set_source(len(self.source))

    def build(self):
        """Extract polygon edges and build reduction matrix."""
        self.edge = BiotFrame(**self.metadata, label='edge', delim='-')
        for ref, poly in enumerate(self.source.poly):
            coords = poly.poly.boundary.xy
            self.edge.insert(ref,
                             coords[0][:-1], coords[0][1:],
                             coords[1][:-1], coords[1][1:],
                             metadata=self.metadata)
        self.edge.set_target(len(self.target))
        for attr in self.edge:
            if attr == 'ref':
                continue
            self.data[attr] = self.edge(attr)
        with self.target_edge():
            attrs = dict(r='x', z='z')
            for attr in attrs:
                self.data[attr] = self.target(attrs[attr])
        self.reduction_matrix = da.zeros((len(self.edge), len(self.source)),
                                         dtype=bool)
        for i in range(len(self.source)):
            self.reduction_matrix[:, i] = self.edge.loc[:, 'ref'] == i
        assert False

    def phi(self, alpha):
        """Return system invariant angle transformation."""
        phi = np.pi - 2*alpha
        if np.isclose(phi, 0, atol=1e-16):
            phi = 1e-16
        return phi

    def beta1(self, alpha):
        """Return beta1 coefficient."""
        phi = self.phi(alpha)
        return (self.r1 - self.r * np.cos(phi))**2 / self.G2(alpha)

    @cached_property
    def gamma(self):
        """Return gamma coefficient."""
        return self.z1 - self.z

    def Gamma(self, alpha):
        """Return Gamma coefficient."""
        phi = self.phi(alpha)
        return self.gamma + self.b1*(self.r1 - self.r*np.cos(phi))

    @cached_property
    def a0(self):
        """Return a0 coefficient."""
        return 1 + self.b1**2

    @cached_property
    def a2(self):
        """Return a2 coefficent."""
        return self.gamma**2 + self.rs

    @cached_property
    def b1(self):
        """Return b1 coefficient."""
        return self.delta_r / self.delta_z

    @cached_property
    def delta_r(self):
        """Return delta r coefficient."""
        return self.r2 - self.r1

    @cached_property
    def delta_z(self):
        """Return delta r coefficient."""
        return self.z2 - self.z1

    def G2(self, alpha):
        """Return G2 coefficient."""
        phi = self.phi(alpha)
        return self.gamma**2 + self.r**2 * np.sin(phi)**2

    def B2(self, alpha):
        """Return B2 coefficient."""
        phi = self.phi(alpha)
        return (self.r1 - self.r * np.cos(phi))**2 + \
            self.a0**2 * self.r**2 * np.sin(phi)**2

    #@cached_property
    def cphi(self, alpha):
        """Return the anti-derivative of Cphi(alpha)."""




if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=-2, dplasma=-150)
    coilset.coil.insert([5, 6], 0.5, 0.2, 0.2, section='h', turn='r',
                        nturn=300, segment='polygon')
    coilset.coil.insert(5.5, 0.5, 0.6, 0.6, section='c', turn='r',
                        nturn=300, segment='polygon')
    coilset.plot()

    coilset.grid.solve(100)
