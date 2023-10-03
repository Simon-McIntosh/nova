"""Biot-Savart calculation for arc segments."""
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.special import ellipj, ellipkinc

from nova.biot.constants import Constants
from nova.biot.matrix import Matrix

# from nova.geometry.rotate import to_vector


@dataclass
class ArcConstants(Constants):
    """Extend Constants class."""

    alpha: ClassVar[float] = np.pi / 2
    num: ClassVar[int] = 120


@dataclass
class Arc(ArcConstants, Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d arc filaments.

    """

    name: ClassVar[str] = "arc"  # element name
    csys: str = "global"

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()

        # use context manager to manage source and target attributes -> unwind on yield
        # define local attributes in local coordinate system
        # map parameters back to global coordinates on solve

        # start_point = self._to_global(self.start_point)
        # print(start_point)
        # transform = np.stack(
        #    [self.source["dx"], self.source["dy"], self.source["dz"]], axis=-1
        # )

        ##Rmat = to_vector(
        #    [0, 0, 1], coilset.subframe.loc[:, ['dx', 'dy', 'dz']]
        # )

        # phi =

        print(self.source("x").shape)

        coords = np.stack(
            [self.source("x"), self.source("y"), self.source("z")], axis=-1
        )
        c_ = np.r_["2,3", self.source("x"), self.source("y"), self.source("z")]
        print("coords", coords.shape, c_.shape)
        # np.einsum('ijk,ijkm->ijm', )

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

    @cached_property
    def Bx(self):
        """Return x-axis alligned field coupling matrix."""
        return self.r[..., 0]


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dwinding=0, field_attrs=["Bx"])
    coilset.winding.insert(
        {"c": (0, 0, 0.5)},
        np.array([[5, 0, 3.2], [0, 5, 3.2], [-5, 0, 3.2]]),
        nturn=2,
    )
    coilset.winding.insert(
        {"c": (0, 0, 0.5)},
        np.array([[-5, 0, 3.2], [0, -5, 3.2], [5, 0, 3.2]]),
        nturn=2,
    )
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[-5, 0, 3.2], [0, 0, 8.2], [5, 0, 3.2]])
    )
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 0, -1.8], [-5, 0, 3.2]])
    )
    coilset.linkframe(["Swp0", "Swp1"])
    coilset.linkframe(["Swp2", "Swp3"])

    coilset.grid.solve(500)

    # coilset.subframe.vtkplot()

    k = 0.3

    theta = np.pi / 3

    u = ellipkinc(theta, k**2)  # Jacobi amplitude

    sn, cn, dn, ph = ellipj(u, k**2)

    print(ph, theta)
