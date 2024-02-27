"""Biot-Savart calculation for rectangular cross-section arc segments."""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.arc import Arc
from nova.biot.matrix import Matrix

from nova.biot.zeta import Zeta


@dataclass
class Bow(Arc, Matrix):
    """
    Extend filament Arc base class.

    Compute interaction for rectangular cross-section arc segments.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "bow"

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.rs = np.stack(
            [self.rs + delta / 2 * self.source("width") for delta in [-1, 1, 1, -1]],
            axis=-1,
        )
        self.zs = np.stack(
            [self.zs + delta / 2 * self.source("height") for delta in [-1, -1, 1, 1]],
            axis=-1,
        )
        for attr in ["r", "z", "alpha", "_phi"]:
            setattr(self, attr, self._stack(attr))

    def _stack(self, attr):
        """Return cross section attribute stacked along last axis."""
        try:
            value = getattr(super(), attr)
        except AttributeError:
            value = getattr(self, attr)
        return np.tile(value[..., np.newaxis], 4)

    @property
    def reps(self):
        """Return tile reps for _pi2 operator."""
        return (len(self.theta), 1, 1, 1)

    @cached_property
    def zeta(self):
        """Return zeta coefficient calculated using jax trapezoid method."""
        return Zeta(self.rs, self.zs, self.r, self.z, self.alpha)()

    @cached_property
    def _Ar_hat(self):
        """Return stacked local radial vector potential intergration coefficents."""
        Ar_hat = (
            self.Cr
            + self.gamma
            * self.a
            / (6 * self.r)
            * self.ellipj["dn"]
            * ((2 * self.rs - self.r) + 2 * self.r * self.ellipj["sn"] ** 2)
            + self.gamma / (6 * self.a * self.r) * self.p_sum(self.Pr, self.Ip)
        )
        return Ar_hat

    @cached_property
    def _Aphi_hat(self):
        """Return stacked local toroidal vector potential intergration coefficents."""
        Aphi_hat = (
            self.Cphi
            + self.gamma * self.r * self.zeta
            + self.gamma
            * self.a
            / (6 * self.r)
            * (
                self.U * self.Kinc
                - 2 * self.rs * self.Einc
                + 2 * self.r * self.ellipj["sn"] * self.ellipj["cn"] * self.ellipj["dn"]
            )
            + self.gamma / (6 * self.a * self.r) * self.p_sum(self.Pphi, self.Pi)
        )
        print(np.prod(self.alpha.shape))
        print(np.sum(self._index))
        return Aphi_hat
        return self._pi2(Aphi_hat)

    def _intergrate(self, data):
        """Return intergral quantity int dalpha dA."""
        data = data[0] - data[1]
        return (
            1
            / (2 * np.pi * self.source("area"))
            * ((data[..., 2] - data[..., 3]) - (data[..., 1] - data[..., 0]))
        )


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 1

    theta = np.linspace(0, 2 * np.pi, 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz"])
    for i in range(segment_number):
        coilset.winding.insert(
            points[2 * i : 1 + 2 * (i + 1)],
            {"rect": (0, 0, 0.05, 0.03)},
            nturn=1,
            minimum_arc_nodes=3,
            Ic=1,
            filament=False,
            ifttt=False,
        )

    coilset.grid.solve(1500, 2)
    coilset.grid.plot("ay", colors="C0")

    cylinder = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz"])
    cylinder.coil.insert(
        {"rect": (radius, height, 0.05, 0.03)}, segment="cylinder", Ic=1
    )
    cylinder.grid.solve(1500, 2)
    cylinder.grid.plot("ay", colors="C2")
    cylinder.plot()
