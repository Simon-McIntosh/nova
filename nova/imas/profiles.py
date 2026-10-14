"""Extract time slices from equilibrium IDS."""

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, ClassVar

import numpy as np

# from scipy.interpolate import griddata, interp1d, RBFInterpolator, RectBivariateSpline
import scipy.interpolate
import scipy.spatial
import contourpy
from matplotlib.path import Path
from scipy.constants import mu_0
from shapely.geometry import Point, Polygon

from nova.biot.plasma import Flux
from nova.imas.database import CoilData
from nova.imas.dataset import Ids

from nova.imas.equilibrium import Equilibrium, EquilibriumData
from nova.imas.getslice import GetSlice
from nova.imas.pf_passive import PF_Passive
from nova.imas.pf_active import PF_Active


@dataclass(frozen=True)
class PrescribedCurrent:
    """One boundary-limited, current-matched toroidal current map."""

    radius: np.ndarray
    height: np.ndarray
    current_density: np.ndarray
    cell_current: np.ndarray
    boundary: np.ndarray
    boundary_source: str
    endpoint_gap: float
    scale_factor: float
    centroid: tuple[float, float]
    plasma_current: float


def finite_flux_interpolant(coordinate, values):
    """Interpolate finite stored flux-function samples through the boundary."""
    coordinate = np.asarray(coordinate, float)
    values = np.asarray(values, float)
    finite = np.isfinite(coordinate) & np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        raise ValueError("flux function has fewer than three finite samples")
    return scipy.interpolate.interp1d(
        coordinate[finite], values[finite], kind="quadratic", fill_value="extrapolate"
    )


def closed_plasma_boundary(radius, height, psi, outline, axis, psi_axis, psi_boundary):
    """Select a simple axis-enclosing outline or its last closed flux surface."""
    radius = np.asarray(radius, float)
    height = np.asarray(height, float)
    outline = np.asarray(outline, float).reshape(-1, 2)
    spacing = min(np.min(np.diff(radius)), np.min(np.diff(height)))
    gap = (
        float(np.linalg.norm(outline[0] - outline[-1]))
        if len(outline)
        else float("nan")
    )

    def valid(points):
        if len(points) < 4 or not np.all(np.isfinite(points)):
            return False
        polygon = Polygon(points)
        return polygon.is_valid and polygon.area > 0 and polygon.contains(Point(axis))

    if np.isfinite(gap) and gap <= 0.1 * spacing and valid(outline):
        return outline, "separatrix", gap

    generator = contourpy.contour_generator(radius, height, np.asarray(psi).T)
    for fraction in (0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8):
        level = psi_axis + fraction * (psi_boundary - psi_axis)
        loops = []
        for points in generator.lines(level):
            if np.linalg.norm(points[0] - points[-1]) <= 0.1 * spacing and valid(
                points
            ):
                loops.append(points)
        if loops:
            return (
                max(loops, key=lambda points: Polygon(points).area),
                "last_closed_flux",
                gap,
            )
    raise ValueError("no simple closed plasma surface encloses the magnetic axis")


def prescribe_current_density(
    radius,
    height,
    psi,
    psi_samples,
    p_prime,
    ff_prime,
    psi_axis,
    psi_boundary,
    plasma_current,
    outline,
    axis,
    flux_derivative_scale=2 * np.pi,
):
    """Evaluate stored flux functions inside the boundary and match current.

    DINA stores derivatives against total poloidal flux. The current-density
    expression uses derivatives against flux per radian, so its gradients are
    multiplied by ``2*pi`` before evaluating the expression.
    """
    radius = np.asarray(radius, float)
    height = np.asarray(height, float)
    psi = np.asarray(psi, float)
    if psi.shape != (len(radius), len(height)) or np.any(radius <= 0):
        raise ValueError("psi must be an (R,Z) map with positive radius")
    if np.isclose(psi_boundary, psi_axis):
        raise ValueError("axis and boundary flux do not define a profile")
    boundary, source, gap = closed_plasma_boundary(
        radius, height, psi, outline, axis, psi_axis, psi_boundary
    )
    grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
    mask = (
        Path(boundary)
        .contains_points(np.c_[grid_r.ravel(), grid_z.ravel()])
        .reshape(psi.shape)
    )
    psi_samples = np.asarray(psi_samples, float)
    coordinate = (psi_samples - psi_samples[0]) / (psi_samples[-1] - psi_samples[0])
    if not np.all(np.diff(coordinate) > 0):
        raise ValueError("the stored flux-function coordinate must increase")
    psi_norm = np.clip((psi - psi_axis) / (psi_boundary - psi_axis), 0, 1)
    pressure = finite_flux_interpolant(coordinate, p_prime)(psi_norm)
    field = finite_flux_interpolant(coordinate, ff_prime)(psi_norm)
    raw_density = np.where(
        mask,
        flux_derivative_scale * (grid_r * pressure + field / (mu_0 * grid_r)),
        0.0,
    )
    cell_area = np.gradient(radius)[:, None] * np.gradient(height)[None, :]
    raw_current = float(np.sum(raw_density * cell_area))
    if not np.isfinite(raw_current) or abs(raw_current) < 1e-12:
        raise ValueError("flux functions integrate to zero or nonfinite current")
    scale = float(plasma_current / raw_current)
    cell_current = raw_density * cell_area * scale
    centroid = (
        float(np.sum(grid_r * cell_current) / plasma_current),
        float(np.sum(grid_z * cell_current) / plasma_current),
    )
    return PrescribedCurrent(
        radius,
        height,
        raw_density * scale,
        cell_current,
        boundary,
        source,
        gap,
        scale,
        centroid,
        float(plasma_current),
    )


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
            return self._interp1d(self.data.psi_norm, self[attr])
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

    def prescribed_current(self, outline: np.ndarray) -> PrescribedCurrent:
        """Prescribe this slice on its stored map from its native flux functions."""
        return prescribe_current_density(
            self["r"],
            self["z"],
            self["psi2d"],
            self["psi1d"],
            self["p_prime"],
            self["ff_prime"],
            self.psi_axis,
            self.psi_boundary,
            self["ip"],
            outline,
            self["magnetic_axis"],
        )

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
        return finite_flux_interpolant(x, y)

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
