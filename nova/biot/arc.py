"""Biot-Savart calculation for arc segments."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.special import ellipj

from nova.biot.constants import Constants
from nova.biot.matrix import Matrix

# from nova.geometry.rotate import to_vector


def arctan2(x1, x2):
    """Return unwraped arctan2 operator."""
    phi = np.arctan2(x1, x2)
    phi[phi < 0] += 2 * np.pi
    return phi


@dataclass
class Arc(Constants, Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d arc elements.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "arc"  # element name

    attrs: dict[str, str] = field(default_factory=lambda: {"dl": "dl"})

    def __post_init__(self):
        """Load source and target geometry in local coordinate system."""
        super().__post_init__()
        self.rs = np.linalg.norm([self("source", "x1"), self("source", "y1")], axis=0)
        self.zs = self("source", "z")
        self.r = np.linalg.norm([self("target", "x"), self("target", "y")], axis=0)
        self.z = self("target", "z")

    @cached_property
    def gamma(self):
        """Return gamma coefficient."""
        return self.zs - self.z
        factor = 1e12
        r2 = self["dl"] ** 2 / 4
        gamma = self.zs - self.z
        a2 = (self.zs - self.z) ** 2 + (self.rs - self.r) ** 2
        return np.where(a2 < r2, gamma * (factor + (1 - factor) * a2 / r2), gamma)

    # @property
    # @offset(1e12)
    # def a2(self):
    #    """Return a**2 coefficient."""
    #    return super().a2

    @cached_property
    def phi(self):
        """Return global target toroidal angle."""
        return arctan2(self.target("y"), self.target("x"))

    @cached_property
    def _phi(self):
        """Return local target toroidal angle."""
        return arctan2(self("target", "y"), self("target", "x"))

    @cached_property
    def alpha(self):
        """Return system invariant angle alpha for start, end, and pi/2."""
        phi_s = np.stack(
            [
                arctan2(self("source", "y1"), self("source", "x1")),
                arctan2(self("source", "y2"), self("source", "x2")),
            ]
        )
        return np.concatenate(
            (
                (np.pi - (phi_s - self._phi[np.newaxis])) / 2,
                np.pi / 2 * np.ones((1,) + self.shape),
            ),
            axis=0,
        )

    @cached_property
    def sign_alpha(self):
        """Return sign(alpha)."""
        return np.sign(self.alpha)

    @cached_property
    def abs_alpha(self):
        """Return abs(alpha)."""
        return abs(self.alpha)

    @cached_property
    def _index(self):
        """Retrun abs alpha > pi/2 segment index."""
        return self.abs_alpha > np.pi / 2

    @cached_property
    def theta(self):
        """Return segment angle."""
        theta = self.abs_alpha.copy()
        theta[self._index] = np.pi - self.abs_alpha[self._index]
        return theta

    @cached_property
    def Kinc(self):
        """Return end point stacked incomplete elliptic intergral of the 1st kind."""
        return np.stack([self.ellipkinc(theta, self.k2) for theta in self.theta])

    @cached_property
    def Einc(self):
        """Return end point stacked incomplete elliptic intergral of the 2nd kind."""
        return np.stack([self.ellipeinc(theta, self.k2) for theta in self.theta])

    @cached_property
    def Winc(self):
        """Return end point stacked composite incomplete elliptic intergral."""
        return np.stack(
            [
                self.Einc[i]
                - self.k2
                * self.ellipj["sn"][i]
                * self.ellipj["cn"][i]
                / self.ellipj["dn"][i]
                for i in range(len(self.theta))
            ]
        )

    @cached_property
    def ellipj(self):
        """Return end point stacked jacobian elliptic functions."""
        return dict(
            zip(
                ["sn", "cn", "dn", "ph"],
                np.stack([ellipj(u, self.k2) for u in self.Kinc], axis=1),
            )
        )

    def _pi2(self, _hat):
        """Index radial and toroidal fields for |alpha| > pi /2."""
        _pi2 = np.tile(_hat[2, np.newaxis], (len(self.theta), 1, 1))
        _hat[self._index] = self.sign_alpha[self._index] * (
            2 * _pi2[self._index] - _hat[self._index]
        )
        return _hat

    @cached_property
    def rack2(self):
        """Return r a ck2 coefficent product."""
        return self.r * self.a * self.ck2

    @property
    def _Ar_hat(self):
        """Return stacked local radial vector potential intergration coefficents."""
        return self.a / self.r * self.ellipj["dn"]

    @property
    def _Aphi_hat(self):
        """Return stacked local toroidal vector potential intergration coefficents."""
        Aphi_hat = (
            self.a
            / self.r
            * self.sign_alpha
            * ((1 - self.k2 / 2) * self.Kinc - self.Einc)
        )
        return self._pi2(Aphi_hat)

    @property
    def _Ax_hat(self):
        """Return stacked local x-coordinate vector potential intergration constants."""
        return self._Ar_hat * np.cos(self._phi) - self._Aphi_hat * np.sin(self._phi)

    @property
    def _Ay_hat(self):
        """Return stacked local y-coordinate vector potential intergration constants."""
        return self._Ar_hat * np.sin(self._phi) + self._Aphi_hat * np.cos(self._phi)

    @property
    def _Az_hat(self):
        """Return stacked local vertical vector potential intergration coefficents."""
        return np.zeros(((len(self.theta),) + self.shape))

    @cached_property
    def _Br_hat(self):
        """Return stacked local radial magnetic field intergration coefficents."""
        Br_hat = (
            self.sign_alpha
            * self.gamma
            * (self.ck2 * self.Kinc - (1 - self.k2 / 2) * self.Winc)
        ) / self.rack2
        return self._pi2(Br_hat)

    @cached_property
    def _Bphi_hat(self):
        """Return stacked local toroidal magnetic field intergration coefficents."""
        return (-self.gamma * self.ck2 / self.ellipj["dn"]) / self.rack2

    @property
    def _Bx_hat(self):
        """Return stacked local x-coordinate magnetic field intergration constants."""
        return self._Br_hat * np.cos(self._phi) - self._Bphi_hat * np.sin(self._phi)

    @property
    def _By_hat(self):
        """Return stacked local y-coordinate magnetic field intergration constants."""
        return self._Br_hat * np.sin(self._phi) + self._Bphi_hat * np.cos(self._phi)

    @property
    def _Bz_hat(self):
        """Return stacked local vertical magnetic field intergration coefficents."""
        Bz_hat = (
            self.sign_alpha
            * (
                self.r * self.ck2 * self.Kinc
                - (self.r - self.b * self.k2 / 2) * self.Winc
            )
        ) / self.rack2
        return self._pi2(Bz_hat)

    def _intergrate(self, data):
        """Return intergral quantity."""
        return self.mu_0 / (4 * np.pi) * (data[0] - data[1])


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 5

    theta = np.linspace(0, 2 * np.pi, 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Bx", "Br", "Bz"])
    for i in range(segment_number):
        coilset.winding.insert(
            points[2 * i : 1 + 2 * (i + 1)],
            {"c": (0, 0, 0.05)},
            nturn=1,
            minimum_arc_nodes=3,
        )

    coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])
    coilset.plot()

    # coilset.subframe.vtkplot()

    coilset.saloc["Ic"] = 5.3e5
    levels = coilset.grid.plot("bz", nulls=False, colors="C2")
    axes = coilset.grid.axes

    circle_coilset = CoilSet(field_attrs=["Br", "Bz", "Aphi"])
    circle_coilset.coil.insert({"c": (radius, height, 0.05, 0.05)})
    circle_coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])
    circle_coilset.saloc["Ic"] = 5.3e5
    circle_coilset.grid.plot(
        "bz", nulls=False, colors="C0", axes=axes, levels=levels, linestyles="--"
    )
