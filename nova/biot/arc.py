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
class Cylinder(ArcConstants, Matrix):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    name: ClassVar[str] = "cylinder"  # element name

    def __post_init__(self):
        """Load intergration constants."""


if __name__ == "__main__":
    k = 0.3

    theta = np.pi / 3

    u = ellipkinc(theta, k**2)  # Jacobi amplitude

    sn, cn, dn, ph = ellipj(u, k**2)

    print(ph, theta)
