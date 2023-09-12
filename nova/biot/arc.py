"""Biot-Savart calculation for arc segments."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.special import ellipj, ellipkinc

from nova.biot.constants import Constants
from nova.biot.matrix import Matrix


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


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dwinding=0, field_attrs=["Bx", "By", "Bz"])
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 5, 3.2], [-5, 0, 3.2]]), nturn=2
    )
    coilset.winding.insert(
        {"c": (0, 0, 0.5)},
        np.array([[-5, 0, 3.2], [0, -5, 3.2], [5, 0, 3.2]]),
        nturn=2,
    )
    coilset.linkframe(["Swp0", "Swp1"])

    coilset.grid.solve(500)

    coilset.plot()

    # coilset.subframe.vtkplot()

    k = 0.3

    theta = np.pi / 3

    u = ellipkinc(theta, k**2)  # Jacobi amplitude

    sn, cn, dn, ph = ellipj(u, k**2)

    print(ph, theta)
