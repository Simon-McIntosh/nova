"""Methods for calculating wall flux position and value."""

from dataclasses import dataclass, field
from functools import cached_property

import jax.numpy as jnp
import xarray

from nova.biot.array import Array
from nova.graphics.plot import Plot
from nova.jax.null import Null1D
from nova.jax.target import Target


@dataclass
class Limiter(Plot, Array):
    """Calculate value and position of limiter wall flux."""

    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    array_attrs: list[str] = field(default_factory=lambda: ["x", "z"])
    data_w: dict[str, float | tuple[float, float]] = field(init=False, repr=False)

    @cached_property
    def null(self):
        """Return jax backed null instance."""
        return Null1D(jnp.c_[self["x"], self["z"]])

    @cached_property
    def target(self):
        """Return jax backed poloidal flux wall target."""
        return Target(
            jnp.array(self.data["Psi"]),
            jnp.array(self.data["Psi_"]),
            self.null,
            source_plasma_index=self.data.source_plasma_index,
        )

    @property
    def w_point(self):
        """Return wall limit point."""
        return self.data_w["point"]

    @property
    def w_psi(self):
        """Return wall limit flux."""
        return self.data_w["psi"]

    def update_wall(self, psi, polarity):
        """Update calculation of field nulls."""
        x_coord, z_coord, psi, _ = self.null(psi, polarity)
        self.data_w = dict(psi=psi, point=(x_coord, z_coord))

    def plot(self, axes=None):
        """Plot null points."""
        self.get_axes("2d", axes)
        self.axes.plot(*self.w_point, "d", ms=4, mec="C3", mew=1, mfc="none")
