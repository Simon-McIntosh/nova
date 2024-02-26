"""Biot-Savart calculation for complete circular cylinders."""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova import njit
from nova.biot.constants import Constants
from nova.biot.matrix import Matrix

from nova.biot.zeta import Zeta


@njit(cache=True)
def zeta(r, rs, z, zs, phi, delta):
    """Return zeta coefficent."""
    result = np.zeros_like(r)
    for i in np.arange(len(phi)):
        result += np.arcsinh(
            (rs - r * np.cos(phi[i]))
            / np.sqrt((zs - z) ** 2 + r**2 * np.sin(phi[i]) ** 2)
        )
    return delta * result


@dataclass
class CylinderConstants(Constants):
    """Extend Constants class."""

    alpha: ClassVar[float] = np.pi / 2
    num: ClassVar[int] = 120

    def Cphi_alpha(self, alpha):
        """Return Cphi(alpha) coefficient."""
        phi = np.pi - 2 * alpha
        return (
            1
            / 2
            * self.gamma
            * self.a
            * np.sqrt(1 - self.k2 * np.sin(alpha) ** 2)
            * -np.sin(2 * alpha)
            - 1
            / 6
            * np.arcsinh(self.beta2(phi))
            * np.sin(2 * alpha)
            * (2 * self.r**2 * np.sin(2 * alpha) ** 2 + 3 * (self.rs**2 - self.r**2))
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.beta1(phi))
            * -np.sin(4 * alpha)
            - 1 / 3 * self.r**2 * np.arctan(self.beta3(phi)) * -np.cos(2 * alpha) ** 3
        )

    @cached_property
    def Cphi(self):
        """Return Cphi intergration constant evaluated between 0 and pi/2."""
        return (
            -1
            / 3
            * self.r**2
            * np.pi
            / 2
            * self.sign(self.gamma)
            * (self.sign(self.rs - self.r) + 1)
        )

    @cached_property
    def _zeta(self):
        """Return zeta coefficient calculated using piecewise-constant."""
        phi, dphi = np.linspace(0, -2 * self.alpha, self.num + 1, retstep=True)
        phi = np.pi + phi[:-1] + dphi / 2
        dalpha = self.alpha / self.num
        return zeta(self.r, self.rs, self.z, self.zs, phi, dalpha)

    @cached_property
    def zeta(self):
        """Return zeta coefficient calculated using jax trapezoid."""
        return Zeta(self.rs, self.zs, self.r, self.z, self.alpha, 50)()

    @property
    def Dz(self):
        """Return Dz coefficient."""
        return 3 / self.r * self.Cphi


@dataclass
class Cylinder(CylinderConstants, Matrix):
    """
    Extend Biot base class.

    Compute interaction for rectangular section complete toroidal conductors.

    """

    axisymmetric: ClassVar[bool] = True
    name: ClassVar[str] = "cylinder"  # element name

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.rs = np.stack(
            [
                self.source("x") + delta / 2 * self.source("dx")
                for delta in [-1, 1, 1, -1]
            ],
            axis=-1,
        )
        self.zs = np.stack(
            [
                self.source("z") + delta / 2 * self.source("dz")
                for delta in [-1, -1, 1, 1]
            ],
            axis=-1,
        )
        self.r = np.stack([self.target("x") for _ in range(4)], axis=-1)
        self.z = np.stack([self.target("z") for _ in range(4)], axis=-1)

    def Aphi_hat(self):
        """Return vector potential intergration coefficient."""
        return (
            self.Cphi
            + self.gamma * self.r * self.zeta
            + self.gamma
            * self.a
            / (6 * self.r)
            * (self.U * self.K - 2 * self.rs * self.E)
            + self.gamma / (6 * self.a * self.r) * self.p_sum(self.Pphi)
        )

    def Br_hat(self):
        """Return radial magnetic field intergration coefficient."""
        return (
            self.r * self.zeta
            - self.a / (2 * self.r) * self.rs * (self.E - self.v * self.K)
            - 1 / (4 * self.a * self.r) * self.p_sum(self.Qr)
        )

    def Bz_hat(self):
        """Return vertical magnetic field intergration coefficient."""
        return (
            self.Dz
            + 2 * self.gamma * self.zeta
            - self.a / (2 * self.r) * 3 / 2 * self.gamma * self.k2 * self.K
            - 1 / (4 * self.a * self.r) * self.p_sum(self.Qz)
        )

    def _intergrate(self, data):
        """Return corner intergration."""
        return (
            1
            / (2 * np.pi * self.source("area"))
            * ((data[..., 2] - data[..., 3]) - (data[..., 1] - data[..., 0]))
        )

    @cached_property
    def Aphi(self):
        """Return Aphi array."""
        return self._intergrate(self.Aphi_hat())

    @property
    def Psi(self):
        """Return Psi array."""
        return 2 * np.pi * self.mu_0 * self.target("x") * self.Aphi

    @cached_property
    def Br(self):
        """Return radial field array."""
        return self.mu_0 * self._intergrate(self.Br_hat())

    @cached_property
    def Bz(self):
        """Return vertical field array."""
        return self.mu_0 * self._intergrate(self.Bz_hat())


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=-1, dplasma=-(15**2))
    """
    coilset.coil.insert(5, 0.5, 0.01, 0.8, segment='cylinder')
    coilset.coil.insert(5.1, 0.5+0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.1, 0.5-0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.2, 0.5, 0.01, 0.8, segment='cylinder')
    """
    coilset.firstwall.insert(
        5.1, 0.52, 0.05, 0.05, turn="s", tile=False, segment="cylinder"
    )

    coilset.saloc["Ic"] = 1

    # coilset.aloc["nturn"] = 0
    # coilset.aloc["nturn"][64] = 1

    coilset.grid.solve(1000, 2.75)
    coilset.grid
    coilset.plot()
    levels = coilset.grid.plot("psi", colors="C0", nulls=False, clabel={})
    # coilset.grid.plot('ke', colors='C0', nulls=False, clabel={})

    coilset = CoilSet(dcoil=-1, dplasma=-(15**2))
    """
    coilset.coil.insert(5, 0.5, 0.01, 0.8, segment='cylinder')
    coilset.coil.insert(5.1, 0.5+0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.1, 0.5-0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.2, 0.5, 0.01, 0.8, segment='cylinder')
    """
    coilset.firstwall.insert(
        5.1, 0.52, 0.05, 0.05, turn="s", tile=False, segment="circle"
    )

    coilset.saloc["Ic"] = 1

    coilset.aloc["nturn"] = 0
    coilset.aloc["nturn"][64] = 1

    coilset.grid.solve(1000, 2.75)

    coilset.grid.plot("psi", colors="C2", nulls=False, clabel={}, levels=levels)
