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

    def __getattr__(self, attr: str):
        """Return attribute, reshape if attribute name has a trailing underscore."""
        match attr[-1]:
            case "_":
                return getattr(self, attr[:-1])[np.newaxis]
            case _:
                raise AttributeError(f"Attribute {attr} not found.")

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
            np.pi / 2 * np.ones((1,) + self.shape),
            axis=0,
        )

    @cached_property
    def sign_alpha(self):
        """Return sign(alpha)."""
        return self.sign(self.alpha)

    @cached_property
    def Kinc(self):
        """Return end point stacked incomplete elliptic intergral of the 1st kind."""
        return np.stack([self.ellipkinc(abs(alpha), self.k2) for alpha in self.alpha])

    @cached_property
    def Einc(self):
        """Return end point stacked incomplete elliptic intergral of the 2nd kind."""
        return np.stack([self.ellipeinc(abs(alpha), self.k2) for alpha in self.alpha])

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

    def Br_hat(self):
        """Return stacked radial magnetic field intergration coefficents."""
        Br_hat = (
            self.sign_alpha
            * self.gamma_
            * (self.ck2_ * self.Kinc - (1 - self.k2_ / 2) * self.Winc)
        )
        index = abs(self.alpha) > np.pi / 2

        _2pi = np.tile(Br_hat[np.newaxis, 2], (3, 1, 1))
        assert np.allclose(_2pi[2], Br_hat[2])
        Br_hat[index] = self.sign_alpha[index] * (2 * _2pi[index] - Br_hat[index])
        print(Br_hat.shape)
        return Br_hat

    def _intergrate(self, data):
        return self.mu_0 / (2 * np.pi**2) * (data[1] - data[0])

    @cached_property
    def _Br(self):
        """Return local radial magnetic field coupling matrix."""
        return self._intergrate(self.Br_hat()) / (self.r * self.a * self.ck2)


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(field_attrs=["Br"])
    coilset.winding.insert(
        np.array([[5, 0, 2], [0, 5, 2], [-5, 0, 2]]),
        {"c": (0, 0, 0.5)},
        nturn=2,
        minimum_arc_nodes=3,
    )
    coilset.winding.insert(
        np.array([[-5, 0, 2], [0, -5, 2], [5, 0, 2]]),
        {"c": (0, 0, 0.5)},
        nturn=2,
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

    coilset.grid.solve(2500, [1, 4.5, 0, 4])

    # coilset.subframe.vtkplot()

    coilset.saloc["Ic"] = 1e3
    levels = coilset.grid.plot("br", nulls=False)
    axes = coilset.grid.axes

    print(coilset.grid.br.max(), coilset.grid.br.min())

    coilset = CoilSet(field_attrs=["Br"])
    coilset.coil.insert({"c": (5, 2, 0.5)})
    coilset.grid.solve(2500, [1, 4.5, 0, 4])
    coilset.saloc["Ic"] = 1e3
    coilset.grid.plot("br", nulls=False, colors="C1", axes=axes, levels=levels)

    print(coilset.grid.br.max(), coilset.grid.br.min())
