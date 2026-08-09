"""Biot-Savart calculation for complete circular cylinders."""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.greens import corner_fields
from nova.biot.matrix import Matrix


@dataclass
class Cylinder(Matrix):
    """
    Extend Biot base class.

    Compute interaction for rectangular section complete toroidal conductors.
    The per-corner antiderivative coefficients come from the canonical
    axisymmetric kernel :func:`nova.biot.greens.corner_fields`; this class
    stacks the section corners against the targets, integrates the four-corner
    definite-integral rule, and normalises per ampere of total conductor
    current.
    """

    axisymmetric: ClassVar[bool] = True
    name: ClassVar[str] = "cylinder"  # element name

    def __post_init__(self):
        """Stack section corners against targets."""
        super().__post_init__()
        self._validate_source_geometry()
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
        self.r = np.stack([self.target("r") for _ in range(4)], axis=-1)
        self.z = np.stack([self.target("z") for _ in range(4)], axis=-1)

    def _validate_source_geometry(self):
        """Reject source sections without finite positive dimensions and area."""
        for attr in ("dx", "dz", "area"):
            values = np.asarray(self.source[attr], dtype=float)
            if np.any(~np.isfinite(values) | (values <= 0.0)):
                raise ValueError(f"cylinder source {attr} must be finite and positive")

    @cached_property
    def _corners(self):
        """Return (Aphi_hat, Br_hat, Bz_hat) per corner from the canonical kernel."""
        return corner_fields(self.rs, self.zs, self.r, self.z)

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
        return self._intergrate(self._corners[0])

    @property
    def Psi(self):
        """Return Psi array."""
        return 2 * np.pi * self.mu_0 * self.target("r") * self.Aphi

    @cached_property
    def Br(self):
        """Return radial field array."""
        return self.mu_0 * self._intergrate(self._corners[1])

    @cached_property
    def Bz(self):
        """Return vertical field array."""
        return self.mu_0 * self._intergrate(self._corners[2])


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(
        dcoil=-1, dplasma=-1, field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz"]
    )  # (15**2)
    """
    coilset.coil.insert(5, 0.5, 0.01, 0.8, segment='cylinder')
    coilset.coil.insert(5.1, 0.5+0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.1, 0.5-0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.2, 0.5, 0.01, 0.8, segment='cylinder')
    """
    coilset.firstwall.insert(
        5.1, 0.52, 0.05, 0.15, turn="r", tile=False, segment="cylinder"
    )

    coilset.saloc["Ic"] = 1

    # coilset.aloc["nturn"] = 0
    # coilset.aloc["nturn"][64] = 1

    coilset.grid.solve(1000, 1.75)
    coilset.plot()
    levels = coilset.grid.plot("bx", colors="C0", nulls=False, clabel={})
    axes = coilset.grid.axes

    coilset = CoilSet(
        dcoil=-1, dplasma=-(50**2), field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz"]
    )
    """
    coilset.coil.insert(5, 0.5, 0.01, 0.8, segment='cylinder')
    coilset.coil.insert(5.1, 0.5+0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.1, 0.5-0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.2, 0.5, 0.01, 0.8, segment='cylinder')
    """
    coilset.firstwall.insert(
        5.1, 0.52, 0.05, 0.15, turn="r", tile=False, segment="circle", ifttt=False
    )
    coilset.plot()
    coilset.saloc["Ic"] = 1

    # coilset.aloc["nturn"] = 0
    # coilset.aloc["nturn"][64] = 1

    coilset.grid.solve(1000, 1.75)
    coilset.grid.plot(
        "bx", colors="C2", nulls=False, clabel={}, levels=levels, axes=axes
    )
