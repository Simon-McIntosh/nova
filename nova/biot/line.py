"""Biot-Savart calculation for line segments."""
from dataclasses import dataclass, field

from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.matrix import Matrix


@dataclass
class Line(Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d line elements.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "line"  # element name

    attrs: dict[str, str] = field(
        default_factory=lambda: {
            "dl": "dl",
            "turnturn": "turnturn",
        }
    )

    @cached_property
    def phi(self):
        """Return global target toroidal angle."""
        return np.arctan2(self.target("y"), self.target("x"))

    @property
    def u2(self):
        """Return stacked u2 coefficient."""
        return np.stack([self("source", "x2") - self("target", "x") for _ in range(2)])

    @property
    def v2(self):
        """Return stacked v2 coefficient."""
        return np.stack([self("source", "y2") - self("target", "y") for _ in range(2)])

    @cached_property
    def wi(self):
        """Return stacked wi coefficient."""
        return np.stack(
            [self("source", f"z{i}") - self("target", "z") for i in range(1, 3)]
        )

    @property
    def _a2(self):
        """Return stacked a2 coefficient."""
        return np.sqrt(self.u2**2 + self.v2**2)

    @cached_property
    def a2(self):
        """Return blended stacked a2 coefficient."""
        r2 = self["dl"] ** 2 / 4
        factor = 1e12
        return np.where((a2 := self._a2) < r2, factor * r2 + a2 * (1 - factor), a2)

    @cached_property
    def ri(self):
        """Return stacked ri coefficient."""
        return np.sqrt(self.a2**2 + self.wi**2)

    @property
    def _Ax_hat(self):
        return np.zeros((2,) + self.shape)

    @property
    def _Ay_hat(self):
        return np.zeros((2,) + self.shape)

    @property
    def _Az_hat(self):
        """Return stacked local z-coord vector potential intergration coefficents."""
        return np.arcsinh(self.wi / self.a2)

    @property
    def _Bx_hat(self):
        """Return stacked local x-coord magnetic field intergration coefficents."""
        return self.wi / (self.ri * self.a2**2) * self.v2

    @property
    def _By_hat(self):
        """Return stacked local y-coord magnetic field intergration coefficents."""
        return self.wi / (self.ri * self.a2**2) * -self.u2

    @property
    def _Bz_hat(self):
        return np.zeros((2,) + self.shape)

    def _intergrate(self, data):
        """Return intergral quantity."""
        return self.mu_0 / (4 * np.pi) * (data[1] - data[0])


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 51

    theta = np.linspace(0, 2 * np.pi, 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Br"])
    coilset.winding.insert(
        points, {"c": (0, 0, 0.5)}, minimum_arc_nodes=len(points) + 1
    )
    coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])

    coilset.saloc["Ic"] = 5.3e5
    levels = coilset.grid.plot("br", nulls=False)
    axes = coilset.grid.axes

    print(coilset.grid.br.max(), coilset.grid.br.min())

    circle_coilset = CoilSet(field_attrs=["Br", "Bz", "Aphi"])
    circle_coilset.coil.insert({"c": (radius, height, 0.05)})
    circle_coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])
    circle_coilset.saloc["Ic"] = 5.3e5
    circle_coilset.grid.plot("br", nulls=False, colors="C1", axes=axes, levels=levels)

    print(circle_coilset.grid.br.max(), circle_coilset.grid.br.min())
