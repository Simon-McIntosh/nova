"""Biot-Savart calculation for arc segments."""

from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import Callable, ClassVar

import numpy as np
import scipy.special

from nova.biot.constants import Constants
from nova.biot.matrix import Matrix


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
        self.zs = self("source", "z1")
        self.r = np.linalg.norm([self("target", "x"), self("target", "y")], axis=0)
        self.z = self("target", "z")

    @cached_property
    def phi(self):
        """Return global target toroidal angle."""
        return np.arctan2(self.target("y"), self.target("x"))

    @cached_property
    def _phi(self):
        """Return local target toroidal angle."""
        return np.arctan2(self("target", "y"), self("target", "x"))

    @cached_property
    def alpha(self):
        """Return system invariant angle alpha for start, end, and pi/2."""
        _phi = self._phi[np.newaxis]
        phi_s = np.stack(
            [
                np.zeros(self.shape, dtype=float),
                arctan2(self("source", "y2"), self("source", "x2")),
            ]
        )
        return np.concatenate(
            (
                (np.pi - (phi_s - _phi)) / 2,
                np.zeros((1,) + self.shape),
                np.pi / 2 * np.ones((1,) + self.shape),
            ),
            axis=0,
        )

    @property
    def sign_alpha(self):
        """Return sign(alpha)."""
        return np.where(self.alpha >= 0, 1, -1)

    @property
    def abs_alpha(self):
        """Return abs(alpha)."""
        return abs(self.alpha)

    def coefficent(func: Callable):
        """Return intergration coefficent evaluated from 0 to theta."""

        @wraps(func)
        def evaluate_coefficent(self):
            result = func(self)
            return result - result[-2]

        return evaluate_coefficent

    @cached_property
    def _index_mask(self):
        """Return index mask."""
        mask = np.ones_like(self.alpha, bool)
        mask[-2:] = False
        return mask

    def mask_index(func: Callable):
        """Return index function with intergral limit mask."""

        @wraps(func)
        def apply_mask(self):
            return func(self) & self._index_mask

        return apply_mask

    @property
    @mask_index
    def _index_A(self):
        """Return |alpha| <= pi/2 segment index."""
        return self.abs_alpha <= np.pi / 2

    @property
    @mask_index
    def _index_B(self):
        """Return |alpha| > pi/2 segment index."""
        return self.abs_alpha > np.pi / 2

    @property
    @mask_index
    def _index_C(self):
        return (self.abs_alpha > np.pi / 2) & (self.abs_alpha <= 3 * np.pi / 2)

    @property
    @mask_index
    def _index_D(self):
        """Return pi/2 < alpha <= 3pi/2 segment index."""
        return (self.alpha > 3 * np.pi / 2) & (self.alpha <= 2 * np.pi)

    @cached_property
    def _theta(self):
        """Return signed segment angle."""
        theta = self.alpha.copy()
        theta[self._index_A] = self.abs_alpha[self._index_A]
        theta[self._index_C] = np.pi - self.abs_alpha[self._index_C]
        theta[self._index_D] = 2 * np.pi - self.alpha[self._index_D]
        return theta

    @property
    def sign_theta(self):
        """Return sign(theta)."""
        return np.where(self._theta >= 0, 1, -1)

    @property
    def theta(self):
        """Return absolute segment angle."""
        return abs(self._theta)

    @cached_property
    def Phi(self):
        """Return system variant angle."""
        phi = np.pi - 2 * self.theta
        sign = np.where(phi >= 0, 1, -1)
        return np.where(sign * phi > 1e4 * self.eps, phi, sign * 1e4 * self.eps)

    @property
    # @coefficent
    def B2(self):
        """Return B2 coefficient."""
        return self.rs**2 + self.r**2 - 2 * self.r * self.rs * np.cos(self.Phi)

    @property
    # @coefficent
    def D2(self):
        """Return D2 coefficient."""
        return self.gamma**2 + self.B2

    @property
    # @coefficent
    def G2(self):
        """Return G2 coefficient."""
        return self.gamma**2 + self.r**2 * np.sin(self.Phi) ** 2

    @property
    # @coefficent
    def beta_1(self):
        """Return beta 1 coefficient."""
        return (self.rs - self.r * np.cos(self.Phi)) / np.sqrt(self.G2)

    @property
    # @coefficent
    def beta_2(self):
        """Return beta 2 coefficient."""
        return self.gamma / np.sqrt(self.B2)

    @property
    # @coefficent
    def beta_3(self):
        """Return beta 3 coefficient."""
        return (
            self.gamma
            * (self.rs - self.r * np.cos(self.Phi))
            / (self.r * np.sin(self.Phi) * np.sqrt(self.D2))
        )

    @cached_property
    # @coefficent
    def Cr(self):
        """Return Cr coefficient."""
        return (
            1
            / 2
            * self.gamma
            * self.a
            * np.sqrt(1 - self.k2 * np.sin(self.theta) ** 2)
            * np.cos(2 * self.theta)
            - 1
            / 6
            * np.arcsinh(self.beta_2)
            * np.cos(2 * self.theta)
            * (
                2 * self.r**2 * np.cos(2 * self.theta) ** 2
                - 3 * (self.rs**2 + self.r**2)
            )
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.beta_1)
            * (3 + np.cos(4 * self.theta))
            - 1 / 3 * self.r**2 * np.arctan(self.beta_3) * np.sin(2 * self.theta) ** 3
        )

    @cached_property
    # @coefficent
    def Cphi(self):
        """Return Cphi coefficient."""
        return (
            1
            / 2
            * self.gamma
            * self.a
            * np.sqrt(1 - self.k2 * np.sin(self.theta) ** 2)
            * -np.sin(2 * self.theta)
            - 1
            / 6
            * np.arcsinh(self.beta_2)
            * np.sin(2 * self.theta)
            * (
                2 * self.r**2 * np.sin(2 * self.theta) ** 2
                + 3 * (self.rs**2 - self.r**2)
            )
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.beta_1)
            * -np.sin(4 * self.theta)
            - 1 / 3 * self.r**2 * np.arctan(self.beta_3) * -np.cos(2 * self.theta) ** 3
        )

    @property
    def reps(self):
        """Return tile reps for _pi2 operator."""
        return (len(self.theta), 1, 1)

    @cached_property
    def rack2(self):
        """Return r a ck2 coefficent product."""
        return self.r * self.a * self.ck2

    @cached_property
    def ellipj(self):
        """Return end point stacked jacobian elliptic functions."""
        return dict(
            zip(
                ["sn", "cn", "dn", "ph"],
                np.stack([scipy.special.ellipj(u, self.k2) for u in self.Kinc], axis=1),
            )
        )

    @cached_property
    def Kinc(self):
        """Return end point stacked incomplete elliptic intergral of the 1st kind."""
        return np.stack([self.ellipkinc(theta, self.k2) for theta in self.theta])

    @cached_property
    def Einc(self):
        """Return end point stacked incomplete elliptic intergral of the 2nd kind."""
        return np.stack([self.ellipeinc(theta, self.k2) for theta in self.theta])

    @cached_property
    def Pi_inc(self) -> dict[int, np.ndarray]:
        """Return end point stacked incomplete elliptic intergral of the 3rd kind."""
        return {
            p: np.stack(
                [self.ellippinc(self.np2[p], theta, self.k2) for theta in self.theta]
            )
            for p in range(1, 4)
        }

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
    def Ip(self) -> dict[int, np.ndarray]:
        """Return I(np2) coefficent."""
        zeros = np.zeros_like(self.k2)
        ones = np.ones_like(self.k2)
        Ip = {p: np.zeros_like(self.k2) for p in range(1, 4)}
        for p in Ip:
            Ip[p] = np.where(
                (sign := self.np2[p] <= 0),
                -np.sqrt(abs(self.np2[p]))
                / (2 * np.sqrt(self.k2 - self.np2[p], ones.copy(), where=sign))
                * np.log(
                    (
                        np.sqrt(self.k2 - self.np2[p], ones.copy(), where=sign)
                        - np.sqrt(abs(self.np2[p])) * self.ellipj["dn"]
                    )
                    ** 2
                    / (1 - self.np2[p] * self.ellipj["sn"] ** 2)
                ),
                Ip[p],
            )
            Ip[p] = np.where(
                (self.np2[p] > 0) & np.isclose(self.k2, self.np2[p]),
                1 / self.ellipj["dn"],
                Ip[p],
            )
            Ip[p] = np.where(
                (sign := self.np2[p] > 0) & (diff := self.np2[p] > self.k2),
                np.sqrt(self.np2[p], zeros.copy(), where=sign)
                / (2 * np.sqrt(self.np2[p] - self.k2, ones.copy(), where=diff))
                * np.log(
                    (
                        np.sqrt(self.np2[p] - self.k2, ones.copy(), where=diff)
                        + np.sqrt(self.np2[p], zeros.copy(), where=sign)
                        * self.ellipj["dn"]
                    )
                    ** 2
                    / (1 - self.np2[p] * self.ellipj["sn"] ** 2)
                ),
                Ip[p],
            )
            Ip[p] = np.where(
                (sign := self.np2[p] > 0) & (diff := self.np2[p] < self.k2),
                -np.sqrt(self.np2[p], zeros.copy(), where=sign)
                / (2 * np.sqrt(self.k2 - self.np2[p], ones.copy(), where=diff))
                * np.arcsin(
                    2
                    * np.sqrt(self.np2[p], zeros.copy(), where=sign)
                    * self.ellipj["dn"]
                    * np.sqrt(self.k2 - self.np2[p], zeros.copy(), where=diff)
                    / (self.k2 * abs(1 - self.np2[p] * self.ellipj["sn"] ** 2))
                ),
                Ip[p],
            )
        return Ip

    def _exterior(self, _hat):
        """Index radial and toroidal fields.

        - pi/2 < |alpha| <= pi
        - pi < alpha <= 3pi/2
        - 3pi/2 < alpha <= 2pi

        """
        _hat_pi2 = np.tile(_hat[-1, np.newaxis], self.reps)
        _hat[self._index_C] = (
            2 * _hat_pi2[self._index_C]
            - self.sign_theta[self._index_C] * _hat[self._index_C]
        )
        _hat[self._index_D] = 4 * _hat_pi2[self._index_D] - _hat[self._index_D]
        return _hat

    @cached_property
    def _Ar_hat(self):
        """Return stacked local radial vector potential intergration coefficents."""
        return self.a / self.r * self.ellipj["dn"]

    @cached_property
    def _Aphi_hat(self):
        """Return stacked local toroidal vector potential intergration coefficents."""
        Aphi_hat = (
            self.a
            / self.r
            * self.sign_theta
            * ((1 - self.k2 / 2) * self.Kinc - self.Einc)
        )
        return self.sign_alpha * self._exterior(Aphi_hat)

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
        """Return stacked local z-coordinate vector potential intergration constants."""
        return np.zeros_like(self._Ar_hat)

    @cached_property
    def _Br_hat(self):
        """Return stacked local radial magnetic field intergration coefficents."""
        Br_hat = (
            self.sign_theta
            * self.gamma
            * (self.ck2 * self.Kinc - (1 - self.k2 / 2) * self.Winc)
        ) / self.rack2
        return self.sign_alpha * self._exterior(Br_hat)

    @cached_property
    def _Bphi_hat(self):
        """Return stacked local toroidal magnetic field intergration coefficents."""
        return (-self.gamma * self.ck2 / self.ellipj["dn"]) / self.rack2

    @property
    def _Bz_hat(self):
        """Return stacked local vertical magnetic field intergration coefficents."""
        Bz_hat = (
            self.sign_theta
            * (
                self.r * self.ck2 * self.Kinc
                - (self.r - self.b * self.k2 / 2) * self.Winc
            )
        ) / self.rack2
        return self.sign_alpha * self._exterior(Bz_hat)

    @property
    def _Bx_hat(self):
        """Return stacked local x-coordinate magnetic field intergration constants."""
        return self._Br_hat * np.cos(self._phi) - self._Bphi_hat * np.sin(self._phi)

    @property
    def _By_hat(self):
        """Return stacked local y-coordinate magnetic field intergration constants."""
        return self._Br_hat * np.sin(self._phi) + self._Bphi_hat * np.cos(self._phi)

    def _intergrate(self, data):
        """Return intergral quantity."""
        return 1 / (4 * np.pi) * (data[0] - data[1])


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 12

    length = 2 * np.pi
    offset = 0

    theta = offset + np.linspace(-length / 2, length / 2, 1 + 3 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Bx", "By", "Br", "Bz", "Ay"])
    coilset.coil.insert(radius, height, 0.05, 0.05, ifttt=False, segment="arc", Ic=1e3)
    """
    for i in range(segment_number):
        coilset.winding.insert(
            points[3 * i : 1 + 3 * (i + 1)],
            {"s": (0, 0, 0.05)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=1e3,
        )
    """
    coilset.grid.solve(1500, 0.5)
    coilset.plot()

    attr = "ay"

    circle = CoilSet(field_attrs=["Bx", "By", "Br", "Bz", "Ay"])
    circle.coil.insert(
        radius, height, 0.05, 0.05, ifttt=False, segment="cylinder", Ic=1e3
    )
    circle.grid.solve(1500, 0.5)
    levels = circle.grid.plot(attr, levels=31, colors="C0", linestyles="--")

    # levels = 31
    coilset.grid.plot(attr, colors="C1", levels=levels)

    """
    # coilset.subframe.vtkplot()

    coilset.saloc["Ic"] = 5.3e5
    levels = coilset.grid.plot("ay", levels=21, nulls=False, colors="C2")
    axes = coilset.grid.axes

    segment_number = 81

    theta = np.linspace(theta[0], theta[-1], 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    line = CoilSet(field_attrs=["Bz", "Ay"])
    for i in range(segment_number):
        line.winding.insert(
            points[2 * i : 1 + 2 * (i + 1)],
            {"s": (0, 0, 0.05)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=5.3e5,
        )

    line.grid.solve(2500, 0.5)
    line.grid.plot("ay", colors="C3", linestyles="--", levels=levels)
    """
