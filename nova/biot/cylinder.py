"""Biot-Savart calculation for complete circular cylinders."""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.greens import MU0, cylinder_greens
from nova.biot.matrix import Matrix


@dataclass
class Cylinder(Matrix):
    """
    Extend Biot base class.

    Compute interaction for rectangular section complete toroidal conductors.
    Every quantity comes from the canonical rectangle kernel
    :func:`nova.biot.greens.cylinder_greens`, which drives the shared corner
    antiderivative over the section's four corners and carries the on-axis
    expansion the corner form cannot reach -- a target ON the symmetry axis
    divides by its own radius inside the antiderivative, so an element that
    stacks the corners itself returns ``nan`` for the whole axis column.
    """

    axisymmetric: ClassVar[bool] = True
    name: ClassVar[str] = "cylinder"  # element name
    mu_0: ClassVar[float] = MU0
    """The kernel's own permeability, so ``Aphi`` inverts ``Psi`` exactly.

    ``Psi`` arrives already carrying the kernel's ``4 pi x 1e-7``.  Dividing it by the
    measured CODATA value the base class holds would leave the two views of one field
    disagreeing by the part in ten billion those constants differ by -- and by a
    different part in ten billion after each CODATA revision, since the measured value
    moves and the defined one does not.  Small, but a fixed offset no rule ever
    removes.  :class:`~nova.biot.circle.Circle` takes the same constant.
    """

    def __post_init__(self):
        """Validate the rectangular source geometry."""
        super().__post_init__()
        self._validate_source_geometry()

    def _validate_source_geometry(self):
        """Reject source sections without finite positive dimensions and area."""
        for attr in ("dx", "dz", "area"):
            values = np.asarray(self.source[attr], dtype=float)
            if np.any(~np.isfinite(values) | (values <= 0.0)):
                raise ValueError(f"cylinder source {attr} must be finite and positive")

    @cached_property
    def _fields(self):
        """Return (Psi, Br, Bz) per ampere from the canonical rectangle kernel."""
        return cylinder_greens(
            self.target("r"),
            self.target("z"),
            self.source("x"),
            self.source("z"),
            self.source("dx"),
            self.source("dz"),
        )

    @cached_property
    def Aphi(self):
        """Return the toroidal vector potential array [Wb/(m.A)].

        ``Phi = 2 pi R A_phi``, inverted here rather than integrated separately so
        the two cannot drift apart.  The axis value is the loop limit ``A_phi = 0``,
        left in place rather than divided for and selected afterwards.
        """
        radius = np.asarray(self.target("r"))
        potential = np.zeros_like(self.Psi)
        np.divide(
            self.Psi,
            2 * np.pi * self.mu_0 * radius,
            out=potential,
            where=radius != 0.0,
        )
        return potential

    @property
    def Psi(self):
        """Return Psi array."""
        return self._fields[0]

    @cached_property
    def Br(self):
        """Return radial field array."""
        return self._fields[1]

    @cached_property
    def Bz(self):
        """Return vertical field array."""
        return self._fields[2]


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
