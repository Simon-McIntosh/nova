"""Biot-Savart calculation for complete circular cylinders."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.constants import Constants
from nova.biot.biotframe import BiotFrame
from nova.biot.matrix import Matrix


@dataclass
class PolygonConstants(Constants):
    """Extend biot constants class to include polygon specific attributes."""

    @cached_property
    def delta_r(self):
        """Return delta rs coefficent."""
        return self.rv2 - self.rv1

    @cached_property
    def delta_z(self):
        """Return delta zs coefficent."""
        return self.zv2 - self.zv1

    @cached_property
    def gamma(self):
        """Return gamma coefficient (override Constant)."""
        return self.zv1 - self.z

    @cached_property
    def b1(self):
        """Return b1 coefficient."""
        return self.delta_r / self.delta_z

    @cached_property
    def r1(self):
        """Return r1 coefficient."""
        return self.rv1 - self.b1

    @cached_property
    def a02(self):
        """Return a0**2 coefficient."""
        return 1 + self.b1**2

    def Gamma(self, phi):
        """Return Gamma coefficient."""
        return self.gamma + self.b1 * (self.rs - self.r * np.cos(phi))

    def G2(self, phi):
        """Return G**2 coefficient (override Constant)."""
        return self.gamma**2 + self.r**2 * np.sin(phi) ** 2

    def B2(self, phi):
        """Return B2 coefficient (override Constant)."""
        return (self.r1 - self.r * np.cos(phi)) ** 2 + self.a02 * self.r**2 * np.sin(
            phi
        ) ** 2

    def D2(self, phi):
        """Return D2 coefficient (override Constant)."""
        return (
            self.gamma**2
            + self.r**2 * np.sin(phi) ** 2
            + (self.rs - self.r * np.cos(phi)) ** 2
        )

    def beta1(self, phi):
        """Return beta1 coefficient."""
        return (self.rs - self.r * np.cos(phi)) / np.sqrt(self.G2(phi))

    def beta2(self, phi):
        """Return beta2 coefficient."""
        return self.Gamma(phi) / np.sqrt(self.B2(phi))

    def beta3(self, phi):
        """Return beta3 coefficient."""
        return (
            self.gamma * (self.rs - self.r * np.cos(phi)) - self.b1 * self.G2(phi)
        ) / (self.r * np.sin(phi) * np.sqrt(self.D2(phi)))

    @cached_property
    def a2(self):
        """Return a2 coefficent."""
        return self.gamma**2 + self.rs

    def cphi(self, alpha):
        """Return the anti-derivative of Cphi(alpha)."""


@dataclass
class Polygon(PolygonConstants, Matrix):
    """
    Extend matrix base class.

    Compute interaction for complete toroidal coils with polygonal sections.

    """

    edge: BiotFrame = field(repr=False, init=False)
    reduction_matrix: np.ndarray = field(repr=False, init=False)

    name: ClassVar[str] = "polygon"  # element name
    attrs: ClassVar[dict[str, str]] = dict(area="area")
    metadata: ClassVar[dict[str, list]] = dict(
        required=["ref", "rv1", "rv2", "zv1", "zv2"],
        additional=[],
        available=[],
        array=["ref"],
        base=[],
    )

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
        self.edge = BiotFrame(**self.metadata, label="edge", delim="-")
        for ref, poly in enumerate(self.source.poly):
            coords = poly.poly.boundary.xy
            self.edge.insert(
                ref,
                coords[0][:-1],
                coords[0][1:],
                coords[1][:-1],
                coords[1][1:],
                metadata=self.metadata,
            )
        self.edge.set_target(len(self.target))
        for attr in self.edge:
            if attr == "ref":
                continue
            self.data[attr] = self.edge(attr)
        with self.target_edge():
            attrs = dict(r="x", z="z")
            for attr in attrs:
                setattr(self, attr, self.target(attrs[attr]))
        self.reduction_matrix = np.zeros((len(self.edge), len(self.source)), dtype=bool)
        for i in range(len(self.source)):
            self.reduction_matrix[:, i] = self.edge.loc[:, "ref"] == i
        self.rs = self.r1 + self.b1 * self.gamma
        #  self.r1 @ self.reduction_matrix).compute()
        print(self.beta1(0).shape)
        assert False


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=-1, nplasma=150)
    coilset.coil.insert(
        [5, 6], 0.5, 0.2, 0.2, section="h", turn="r", nturn=300, segment="polygon"
    )
    coilset.coil.insert(
        5.5, 0.5, 0.6, 0.6, section="c", turn="r", nturn=300, segment="polygon"
    )
    coilset.plot()

    coilset.grid.solve(100)
