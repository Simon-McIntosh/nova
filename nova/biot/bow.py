"""Biot-Savart calculation for rectangular cross-section arc segments."""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.arc import Arc
from nova.biot.matrix import Matrix

from nova.biot.constants import Zeta


@dataclass
class Bow(Arc, Matrix):
    """
    Extend Biot base class.

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
        self.r = np.stack([self.r for _ in range(4)], axis=-1)
        self.z = np.stack([self.z for _ in range(4)], axis=-1)

        print(self.alpha.shape)

        print(Zeta(self.rs, self.zs, self.r, self.z, self.alpha).gamma)

    def _stack(self, attr, axis=0):
        """Return cross section attribute stacked along axis."""
        return np.stack([getattr(super(), attr) for _ in range(4)], axis=axis)

    @cached_property
    def alpha(self):
        """Return stacked system invariant angle."""
        return self._stack("alpha", -1)

    @cached_property
    def _phi(self):
        """Return stacked local target toroidal angle."""
        return self._stack("_phi", -1)

    @property
    def reps(self):
        """Return tile reps for _pi2 operator."""
        return (len(self.theta), 1, 1, 1)

    @property
    def _Az_hat(self):
        """Return stacked local vertical vector potential intergration coefficents."""
        return self._stack("_Az_hat", -1)

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
            {"annulus": (0, 0, 0.05, 0.2)},
            nturn=1,
            minimum_arc_nodes=3,
            Ic=1,
            filament=False,
        )

    coilset.grid.solve(2500, [2, 3.8, 1, 3])
    coilset.grid.plot("ay")
    coilset.plot()
