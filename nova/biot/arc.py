"""Biot-Savart calculation for arc segments."""
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.special import ellipj

from nova.biot.constants import Constants
from nova.biot.matrix import Matrix

# from nova.geometry.rotate import to_vector


@dataclass
class Arc(Constants, Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d arc filaments.

    """

    name: ClassVar[str] = "arc"  # element name

    def __post_init__(self):
        """Load source and target geometry in local coordinate system."""
        super().__post_init__()
        self.rs = np.linalg.norm([self("source", "x1"), self("source", "y1")], axis=0)
        self.zs = self("source", "z")
        self.r = np.linalg.norm([self("target", "x"), self("target", "y")], axis=0)
        self.z = self("target", "z")
        self.phi = np.arctan2(self("target", "y"), self("target", "x"))

    '''
    def __getattr__(self, attr: str):
        """Return attribute, reshape if attribute name has a trailing underscore."""
        match attr[-1]:
            case "_":
                return getattr(self, attr[:-1])[np.newaxis]
            case _:
                raise AttributeError(f"Attribute {attr} not found.")
    '''

    @cached_property
    def alpha(self):
        """Return system invariant angle alpha for start, end, and pi/2."""
        phi_s = np.stack(
            [
                np.arctan2(self("source", "y1"), self("source", "x1")),
                np.arctan2(self("source", "y2"), self("source", "x2")),
            ]
        )
        return np.append(
            (np.pi - (phi_s[:2] - self.phi[np.newaxis, :])) / 2,
            np.zeros((1,) + self.shape),
            np.pi / 2 * np.ones((1,) + self.shape),
            axis=0,
        )

    @cached_property
    def sign_alpha(self):
        """Return sign(alpha)."""
        return self.sign(self.alpha)

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
                for i in range(3)
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

    def _Bpi2(self, B_hat):
        """Index radial and toroidal magnetic fields for abs alpha > pi /2."""
        _pi2 = np.tile(B_hat[np.newaxis, 3], (3, 1, 1))
        B_hat[self._index] = self.sign_alpha[self._index] * (
            2 * _pi2[self._index] - B_hat[self._index]
        )
        return B_hat

    @cached_property
    def rack2(self):
        """Return r a ck2 coefficent product."""
        return self.r * self.a * self.ck2

    @property
    def _Br_hat(self):
        """Return stacked local radial magnetic field intergration coefficents."""
        Br_hat = (
            self.sign_alpha
            * self.gamma
            * (self.ck2 * self.Kinc - (1 - self.k2 / 2) * self.Winc)
        ) / self.rack2
        return self._Bpi2(Br_hat)

    @property
    def _Bphi_hat(self):
        """Return stacked local toroidal magnetic field intergration coefficents."""
        return (-self.gamma * self.ck2 / self.ellipj["dn"]) / self.rack2

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
        return self._Bpi2(Bz_hat)

    def _intergrate(self, data):
        return self.mu_0 / (np.pi**2) * (data[0] - data[1])

    @cached_property
    def _Bcyl(self):
        """Return local magnetic field in cylindrical coordinates."""
        return np.stack(
            [
                self._intergrate(getattr(self, f"_B{attr}_hat"))
                for attr in ["r", "phi", "z"]
            ],
            axis=0,
        )

    @property
    def _Bcart(self):
        """Return local magnetic field vector in cartesian frame."""
        return np.stack(
            [
                self._Bcyl[0] * np.cos(self.phi) - self._Bcyl[1] * np.sin(self.phi),
                self._Bcyl[0] * np.sin(self.phi) + self._Bcyl[1] * np.cos(self.phi),
                self._Bcyl[2],
            ],
            axis=-1,
        )

    @cached_property
    def B_cart(self):
        """Retrun global magnetic field vector."""
        return self.loc.rotate(self._Bcart, "to_global")

    @property
    def Br(self):
        """Return radial field component."""
        return self.B_cart[..., 0]


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 13

    theta = np.linspace(0, 2 * np.pi, 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )
    print(points.shape)

    coilset = CoilSet(field_attrs=["Br"])
    for i in range(segment_number):
        coilset.winding.insert(
            points[2 * i : 1 + 2 * (i + 1)],
            {"c": (0, 0, 0.5)},
            nturn=1,
            minimum_arc_nodes=3,
        )

    """
    coilset.winding.insert(
        np.array([[-5, 0, 2], [0, 0, 8.2], [5, 0, 2]]), {"c": (0, 0, 0.5)}
    )
    coilset.winding.insert(
        np.array([[5, 0, 2], [0, 0, -1.8], [-5, 0, 2]]), {"c": (0, 0, 0.5)}
    )
    """
    coilset.linkframe(["Swp0", "Swp1"])
    # coilset.linkframe(["Swp2", "Swp3"])

    coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])

    # coilset.subframe.vtkplot()

    coilset.saloc["Ic"] = 1e3
    levels = coilset.grid.plot("br", nulls=False)
    axes = coilset.grid.axes

    print(coilset.grid.br.max(), coilset.grid.br.min())

    circle_coilset = CoilSet(field_attrs=["Br", "Bz"])
    circle_coilset.coil.insert({"c": (radius, 2, 0.5)})
    circle_coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])
    circle_coilset.saloc["Ic"] = 1e3
    circle_coilset.grid.plot("br", nulls=False, colors="C1", axes=axes, levels=levels)

    print(circle_coilset.grid.br.max(), circle_coilset.grid.br.min())
