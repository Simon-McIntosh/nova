"""Methods for calculating wall flux position and value."""

from dataclasses import dataclass, field
from functools import cached_property

import xarray

from nova.biot.array import Array
from nova.geometry import select
from nova.graphics.plot import Plot


@dataclass
class Limiter(Plot, Array):
    """Calculate value and position of limiter wall flux."""

    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    array_attrs: list[str] = field(default_factory=lambda: ["x", "z"])
    data_w: dict[str, float | tuple[float, float]] = field(init=False, repr=False)

    @cached_property
    def null(self):
        """Return jax backed null instance."""
        import jax.numpy as jnp  # noqa: PLC0415

        from nova.jax.null import Null1D  # noqa: PLC0415

        return Null1D(jnp.c_[self["x"], self["z"]])

    @cached_property
    def target(self):
        """Return jax backed poloidal flux wall target."""
        import jax.numpy as jnp  # noqa: PLC0415

        from nova.jax.target import Target  # noqa: PLC0415

        return Target(
            jnp.array(self.data["Psi"]),
            jnp.array(self.data["Psi_"]),
            self.null,
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
        """Publish the wall-limit point and flux."""
        x_coord, z_coord, psi, _ = select.wall_flux(self["x"], self["z"], psi, polarity)
        self.data_w = dict(psi=psi, point=(x_coord, z_coord))

    def plot(self, axes=None):
        """Plot null points."""
        self.get_axes("2d", axes)
        self.axes.plot(*self.w_point, "d", ms=4, mec="C3", mew=1, mfc="none")
