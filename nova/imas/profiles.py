"""Extract time slices from equilibrium IDS."""

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, ClassVar

import jax.numpy as jnp
import numpy as np

# from scipy.interpolate import griddata, interp1d, RBFInterpolator, RectBivariateSpline
import scipy.interpolate
import scipy.spatial

from nova.biot.plasma import Flux
from nova.imas.database import CoilData
from nova.imas.dataset import Ids

from nova.imas.equilibrium import Equilibrium, EquilibriumData
from nova.imas.getslice import GetSlice
from nova.imas.pf_passive import PF_Passive
from nova.imas.pf_active import PF_Active
from nova.jax.basis import Interp


@dataclass
class Profile(Flux, Equilibrium, GetSlice, CoilData):
    """Interpolation of profiles from an equilibrium time slice."""

    equilibrium: Ids | bool | str = True
    pf_active: Ids | bool | str = False
    pf_passive: Ids | bool | str = False

    dataset: ClassVar[dict] = {
        "equilibrium": EquilibriumData,
        "pf_active": PF_Active,
        "pf_passive": PF_Passive,
    }

    def __post_init__(self):
        """Build and merge ids datasets."""
        for attr, IdsClass in self.dataset.items():
            if isinstance(ids_attrs := getattr(self, attr), str):
                continue
            setattr(self, attr, self.get_ids_attrs(ids_attrs, IdsClass))
        super().__post_init__()

    def build(self):
        """Merge dataset data."""
        super().build()
        for IdsClass in self.dataset.values():
            self.load_data(IdsClass)

    @property
    def _dataset_attrs(self) -> list[str]:
        """Extend Machine dataset attributes."""
        attrs = super()._dataset_attrs
        return attrs + [attr for attr in self.dataset if attr not in attrs]

    def fluxfunctions(self, attr) -> Callable:
        """Retrun flux function interpolant for attr."""
        if attr in ["p_prime", "ff_prime"]:
            # return self._interp1d(self.data.psi_norm, self[attr])
            return Interp(jnp.array(self.data.psi_norm)) / jnp.array(self[attr])
        return super().fluxfunctions(attr)

    @property
    def psi_axis(self):
        """Return on-axis poloidal flux."""
        return float(self["psi_axis"])

    @property
    def psi_boundary(self):
        """Return boundary poloidal flux."""
        return float(self["psi_boundary"])

    def normalize(self, psi):
        """Return normalized poloidal flux."""
        return (psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    def denormalize(self, psi_norm):
        """Return poloidal flux."""
        return psi_norm * (self.psi_boundary - self.psi_axis) + self.psi_axis

    @property
    def psi_rbs(self):
        """Return cached 2D RectBivariateSpline psi2d interpolant."""
        return self._rbs("psi2d")

    @property
    def j_tor_rbs(self):
        """Return cached 2D RectBivariateSpline j_tor2d interpolant."""
        return self._rbs("j_tor2d")

    @cached_property
    def delaunay(self):
        """Return Delaunay triangulation of unstructured equilibrium grid."""
        return scipy.spatial.Delaunay(np.c_[self.data.r2d, self.data.z2d])

    def _rbs(self, attr):
        """Return 2D RectBivariateSpline interpolant."""
        try:
            return scipy.interpolate.RectBivariateSpline(
                self["r"], self["z"], self[attr]
            ).ev
        except KeyError:
            return scipy.interpolate.LinearNDInterpolator(self.delaunay, self[attr])
            # return lambda radius, height: griddata(
            #    np.c_[self.data.r2d, self.data.z2d], self[attr], np.c_[radius, height]
            # )
            # return lambda radius, height: RBFInterpolator(
            #    np.c_[self.data.r2d, self.data.z2d], self[attr]
            # )(np.c_[radius, height])

    def _interp1d(self, x, y):
        """Return 1D interpolant."""
        return scipy.interpolate.interp1d(
            x, y, kind="quadratic", fill_value="extrapolate"
        )

    def plot_profile(self, attr="p_prime", axes=None):
        """Plot flux function interpolants."""
        self.set_axes("1d", axes=axes)
        psi_norm = np.linspace(0, 1, 500)
        self.axes.plot(self.data.psi_norm, self[attr], ".")
        self.axes.plot(psi_norm, getattr(self, attr)(psi_norm), "-")
        self.axes.set_ylabel(attr)
        self.axes.set_xlabel(r"$\psi_{norm}$")

    def plot_current(self, axes=None):
        """Plot current timeseries."""
        self.set_axes("1d", axes=axes)
        self.axes.plot(self.data.time, 1e-3 * self.data.current)
        self.axes.set_xlabel("time s")
        self.axes.set_ylabel(r"$I$ kA")


if __name__ == "__main__":
    pulse, run = 135013, 2
    profile = Profile(pulse, run, machine="iter", equilibrium=True)
    profile.time = 300
    profile.plot_profile(attr="ff_prime")
