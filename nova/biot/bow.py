"""Biot-Savart calculation for rectangular cross-section arc segments."""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.arc import Arc
from nova.biot.matrix import Matrix

from nova.biot.zeta import zeta


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

    @cached_property
    def Dr(self):
        """Return radial D coefficient."""
        return self.r / 4 * np.sin(4 * self.theta) * np.arcsinh(self.beta_1)

    @cached_property
    def Dphi(self):
        """Return toroidal D coefficient."""
        return self.r / 4 * np.cos(4 * self.theta) * np.arcsinh(self.beta_1)

    @cached_property
    def Dz(self):
        """Return vertical D coefficient."""
        return self.r * (
            np.sin(2 * self.theta) * np.arcsinh(self.beta_2)
            + np.cos(2 * self.theta) * np.arctan(self.beta_3)
        )

    @cached_property
    def zeta(self):
        """Return zeta coefficient calculated using numba's trapezoid method."""
        return zeta(
            np.tile(self.rs, self.reps),
            np.tile(self.r, self.reps),
            np.tile(self.gamma, self.reps),
            self.theta,
        )

    @property
    def reps(self):
        """Return tile reps for _pi2 operator."""
        return (len(self.theta), 1, 1, 1)

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
            + self.gamma / (6 * self.a * self.r) * self.p_sum(self.Pphi, self.Pi_inc)
        )
        return self.sign_alpha * self._exterior(Aphi_hat)

    @cached_property
    def _Br_hat(self):
        """Return stacked local radial magnetic field intergration coefficents."""
        Br_hat = (
            self.Dr
            + self.r * self.zeta
            - self.a
            / (2 * self.r)
            * (
                self.rs * (self.Einc - self.v * self.Kinc)
                + 2 * self.r * self.ellipj["sn"] * self.ellipj["cn"] * self.ellipj["dn"]
            )
            - 1 / (4 * self.a * self.r) * self.p_sum(self.Qr, self.Pi_inc)
        )
        return self.sign_alpha * self._exterior(Br_hat)

    @cached_property
    def _Bphi_hat(self):
        """Return stacked local toroidal magnetic field intergration coefficents."""
        return (
            self.Dphi
            - self.a
            / (2 * self.r)
            * self.ellipj["dn"]
            * (self.b - 2 * self.r * self.ellipj["sn"] ** 2)
            - 1 / (4 * self.a * self.r) * self.p_sum(self.Qphi, self.Ip)
        )

    @cached_property
    def _Bz_hat(self):
        """Return stacked local vertical magnetic field intergration coefficents."""
        Bz_hat = (
            self.Dz
            + 2 * self.gamma * self.zeta
            - self.a / (2 * self.r) * 3 / 2 * self.gamma * self.k2 * self.Kinc
            - 1 / (4 * self.a * self.r) * self.p_sum(self.Qz, self.Pi_inc)
        )
        return self.sign_alpha * self._exterior(Bz_hat)

    def _intergrate(self, data):
        """Return intergral quantity int dalpha dA."""
        data = data[0] - data[1]
        print("bow", self.source("area").ravel()[0])
        return (
            1
            / (4 * np.pi * self.source("area"))
            * ((data[..., 2] - data[..., 1]) - (data[..., 3] - data[..., 0]))
        )


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 1

    length = 2 * np.pi
    offset = np.pi * 0 + 0.5

    theta = offset + np.linspace(-length / 2, length / 2, 1 + 3 * segment_number)

    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    attr = "ay"
    factor = 0.5

    bow = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz", "Br"])
    for i in range(segment_number):
        bow.winding.insert(
            points[3 * i : 1 + 3 * (i + 1)],
            {"rect": (0, 0, 0.06, 0.03)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=1,
            filament=False,
            ifttt=False,
        )

    print("bow solve")
    bow.grid.solve(1500, factor)

    arc = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz", "Br"])
    for i in range(segment_number):
        arc.winding.insert(
            points[3 * i : 1 + 3 * (i + 1)],
            {"rect": (0, 0, 0.06, 0.03)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=1,
            filament=True,
            ifttt=False,
        )

    print("arc solve")
    arc.grid.solve(1500, factor)

    line = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz", "Br"])
    for i in range(segment_number):
        line.winding.insert(
            points[2 * i : 1 + 2 * (i + 1)],
            {"rect": (0, 0, 0.06, 0.03)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=1,
            filament=True,
            ifttt=False,
        )
    line.grid.solve(1500, factor)

    cylinder = CoilSet(field_attrs=["Bx", "By", "Bz", "Ay", "Br"])
    cylinder.coil.insert(
        radius, height, 0.06, 0.03, ifttt=False, segment="cylinder", Ic=1
    )
    cylinder.grid.solve(1500, factor)
    levels = cylinder.grid.plot(attr, levels=31, colors="C3", linestyles="-")

    # assert np.allclose(getattr(cylinder.grid, attr), getattr(bow.grid, attr),
    #                   atol=1e-2)

    arc.plot()
    levels = bow.grid.plot(attr, colors="C0", linestyles="--", levels=levels)
    # arc.grid.plot(attr, colors="C2", linestyles="-.", levels=levels)

    # line.grid.plot(attr, colors="C1", linestyles="--", levels=levels)
