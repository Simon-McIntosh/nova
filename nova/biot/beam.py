"""Biot-Savart calculation for line segments."""

from dataclasses import dataclass, field

from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.matrix import Matrix


@dataclass
class Beam(Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d line elements with a finite cross-section.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "beam"  # element name

    attrs: dict[str, str] = field(default_factory=lambda: {"dl": "dl"})

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.xs = np.stack(
            [
                self("source", "x2") + delta / 2 * self.source("width")
                for delta in [-1, 1]
            ],
        )
        self.ys = np.stack(
            [
                self("source", "y2") + delta / 2 * self.source("height")
                for delta in [-1, 1]
            ],
        )
        self.zs = np.stack([self("source", "z1"), self("source", "z2")])
        self.x = self("target", "x")
        self.y = self("target", "y")
        self.z = self("target", "z")

    @property
    def ui(self):
        """Return ui coefficent."""
        return (self.xs - self.x[np.newaxis])[:, np.newaxis, np.newaxis]

    @property
    def vj(self):
        """Return vi coefficent."""
        return (self.ys - self.y[np.newaxis])[np.newaxis, :, np.newaxis]

    @property
    def wk(self):
        """Return wi coefficent."""
        return (self.zs - self.z[np.newaxis])[np.newaxis, np.newaxis]

    @cached_property
    def alpha(self):
        """Return alpha_ijk coefficent."""
        return np.sqrt(self.ui**2 + self.vj**2)

    @cached_property
    def beta(self):
        """Return beta_ijk coefficent."""
        return np.sqrt(self.vj**2 + self.wk**2)

    @cached_property
    def gamma(self):
        """Return gamma_ijk coefficent."""
        return np.sqrt(self.wk**2 + self.ui**2)

    @cached_property
    def theta(self) -> dict[str, np.ndarray]:
        """Return theta coefficents 1-6."""
        r = np.sqrt(self.ui**2 + self.vj**2 + self.wk**2)
        return dict(
            zip(
                np.arange(1, 7),
                [
                    self.wk / self.alpha,
                    self.ui / self.beta,
                    self.vj / self.gamma,
                    self.vj * self.wk / (self.ui * r),
                    self.wk * self.ui / (self.vj * r),
                    self.ui * self.vj / (self.wk * r),
                ],
            )
        )

    @cached_property
    def phi(self):
        """Return global target toroidal angle."""
        return np.arctan2(self.target("y"), self.target("x"))

    @property
    def _Ax_hat(self):
        return np.zeros((2, 2, 2) + self.shape)

    @property
    def _Ay_hat(self):
        return np.zeros((2, 2, 2) + self.shape)

    @property
    def _Az_hat(self):
        """Return stacked local z-coord vector potential intergration coefficents."""
        return (
            self.ui * self.vj * np.arcsinh(self.theta[1])
            + self.vj * self.wk * np.arcsinh(self.theta[2])
            + self.wk * self.ui * np.arcsinh(self.theta[3])
            - 0.5
            * (
                self.ui**2 * np.arctan(self.theta[4])
                + self.vj**2 * np.arctan(self.theta[5])
                + self.wk**2 * np.arctan(self.theta[6])
            )
        )

    @property
    def _Bx_hat(self):
        """Return stacked local x-coord magnetic field intergration coefficents."""
        return (
            -self.ui * np.arcsinh(self.theta[1])
            - self.wk * np.arcsinh(self.theta[2])
            + self.vj * np.arctan(self.theta[5])
        )

    @property
    def _By_hat(self):
        """Return stacked local y-coord magnetic field intergration coefficents."""
        return (
            self.vj * np.arcsinh(self.theta[1])
            + self.wk * np.arcsinh(self.theta[3])
            - self.ui * np.arctan(self.theta[4])
        )

    @property
    def _Bz_hat(self):
        return np.zeros((2, 2, 2) + self.shape)

    @cached_property
    def _sign(self):
        """Return intergrator sign."""
        i = j = k = np.arange(1, 3)
        return (-1) ** (
            i[:, np.newaxis, np.newaxis]
            + j[np.newaxis, :, np.newaxis]
            + k[np.newaxis, np.newaxis]
        )

    def _intergrate(self, data):
        """Return intergral quantity."""
        return (
            1
            / (4 * np.pi * self.source("area"))
            * np.einsum("ijk,ijk...", self._sign, data)
        )


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 4

    attr = "ay"
    factor = 0.3
    Ic = 5.3e5

    outer_width = 0.05
    inner_width = 0.04

    theta = np.linspace(0, 2 * np.pi, segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Ay", "Bx", "By", "Bz", "Br"])
    coilset.winding.insert(
        points,
        {"box": (0, 0, outer_width, 1 - inner_width / outer_width)},
        minimum_arc_nodes=4,
        filament=False,
        ifttt=False,
    )

    coilset.plot()

    coilset.point.solve(np.array([radius, height]))

    coilset.grid.solve(2500, factor)

    coilset.saloc["Ic"] = Ic

    levels = coilset.grid.plot(attr, colors="C0", levels=61)

    axes = coilset.grid.axes

    cylinder = CoilSet(field_attrs=["Ay", "Bx", "By", "Bz", "Br"])
    cylinder.coil.insert({"rect": (radius, height, outer_width, outer_width)})
    cylinder.coil.insert({"rect": (radius, height, inner_width, inner_width)})
    # cylinder.linkframe(cylinder.frame.index, -1)

    Ashell = outer_width**2 - inner_width**2
    Jc = Ic / Ashell
    cylinder.grid.solve(2500, factor)
    cylinder.saloc["Ic"] = Jc * outer_width**2, -Jc * inner_width**2

    levels = cylinder.grid.plot(
        attr, levels=levels, colors="C1", axes=axes, linestyles="--"
    )

    # cylinder.plot()
