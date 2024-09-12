"""Forward free-boundary equilibrium solver."""

from contextlib import contextmanager
from dataclasses import dataclass, field, InitVar
from functools import cached_property

import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import interp1d
from scipy.optimize import newton_krylov
import scipy.spatial

from nova.database.netcdf import netCDF
from nova.biot.flux import Flux
from nova.biot.levelset import LevelSet
from nova.biot.plasmagrid import PlasmaGrid
from nova.biot.plasmawall import PlasmaWall
from nova.frame.plasmaloc import PlasmaLoc
from nova.geometry.curve import LCFS
from nova.geometry.polygon import Polygon
from nova.geometry.strike import Strike
from nova.graphics.plot import Plot


@dataclass
class Plasma(Plot, netCDF, Flux, PlasmaLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = "plasma"
    grid: PlasmaGrid = field(repr=False, default_factory=PlasmaGrid)
    wall: PlasmaWall = field(repr=False, default_factory=PlasmaWall)
    levelset: LevelSet = field(repr=False, default_factory=LevelSet)
    lcfs: LCFS | None = field(init=False, repr=False, default=None)
    fluxfunctions: InitVar[dict] = field(repr=False, default=None)

    def __post_init__(self, fluxfunctions):
        """Link flux functions and update subframe metadata."""
        self.fluxfunctions = fluxfunctions
        self.subframe.metaframe.metadata = {
            "additional": ["plasma", "ionize", "area", "nturn"],
            "array": ["plasma", "ionize", "area", "nturn", "x", "z"],
        }
        self.subframe.update_columns()
        super().__post_init__()
        self.version["lcfs"] = None
        self.version["wall"] = None

    @cached_property
    def strike(self):
        """Return strike instance."""
        return Strike()

    def __getattribute__(self, attr):
        """Extend getattribute to intercept grid and wall data access."""
        match attr:
            case "lcfs":
                self._check_lcfs()
            # case "w_point" | "psi_w":
            #    self._check_wall()
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            return getattr(self.lcfs, attr)

    def solve(self, boundary=None):
        """Solve interaction matricies across plasma grid."""
        self.wall.solve(boundary=boundary)
        self.grid.solve()
        self.levelset.solve()

    def update_lcfs(self):
        """Update last closed flux surface."""
        if len(self.levelset) == 0:
            raise RuntimeError("solve levelset - nlevelset is None")
        points = self.levelset(self.psi_lcfs)
        mask = self.grid.x_mask(points[:, 1])
        self.lcfs = LCFS(points[mask])

    def _check_lcfs(self):
        """Check validity of upstream data, update psi if nessisary."""
        self.levelset.check_plasma("Psi")
        if (
            self.version["lcfs"] is None
            or self.levelset.version["psi"]
            or self.version["lcfs"] != self.levelset.version["psi"]
        ):
            self.update_lcfs()
            self.version["lcfs"] = self.levelset.version["psi"]

    def update_wall(self):
        """Update wall mask."""
        o_point = self.grid["o_point"]
        for x_point in self.grid.x_points[: self.grid.x_point_number]:
            if x_point[1] < o_point[1]:
                self.wall["psi"] = np.where(
                    self.wall["z"] < x_point[1], np.nan, self.wall["psi"]
                )
            # if x_point[1] > o_point[1]:
            #    self.wall["psi"] = np.where(
            #        self.wall["z"] > x_point[1], np.nan, self.wall["psi"]
            #    )
        self.wall.version["limitflux"] = None

    def _check_wall(self):
        """Check validity of wall mask, update mask if nessisary."""
        if (
            self.version["wall"] is None
            or self.version["wall"] != self.wall.version["psi"]
        ):
            self.update_wall()
            self.version["wall"] = self.wall.version["psi"]

    @property
    def li_3(self):
        """Return normalized plasma inductance."""
        filament_volume = self.aloc["ionize", "volume"]
        volume = np.sum(filament_volume)
        poloidal_field = self.grid.bp[self.aloc["plasma", "ionize"]]
        surface = np.sum(poloidal_field**2 * filament_volume) / volume
        # boundary = (mu_0 * self.i_plasma / self.lcfs.length)**2
        radius = 6.2  # self.lcfs.geometric_radius
        boundary = (mu_0 * self.i_plasma) ** 2 * radius / (2 * volume)
        return surface / boundary

    @cached_property
    def psi_index(self):
        """Return plasma / wall number."""
        return np.cumsum([self.grid.number, self.wall.number])

    @property
    def psi_axis(self):
        """Return on-axis poloidal flux."""
        return self.grid["o_psi"]

    @property
    def magnetic_axis(self):
        """Return location of plasma o-point."""
        return self.grid["o_point"]

    @property
    def psi_x(self):
        """Return primary x-point poloidal flux."""
        return self.grid["x_psi"]

    @property
    def x_point(self):
        """Return location of primary x-point."""
        return self.grid["x_point"]

    @property
    def psi_w(self):
        """Return wall limiter poloidal flux."""
        return self.wall["w_psi"]

    @property
    def w_point(self):
        """Return wall limiter poloidal flux."""
        return self.wall["w_point"]

    @property
    def limiter(self):
        """Return True if plasma is in a limter state."""
        return np.isclose(self.psi_boundary, self.psi_w)

    @property
    def psi_boundary(self):
        """Return boundary poloidal flux."""
        if self.grid.x_point_number == 0:
            return self.psi_w
        x_height = self.grid.x_points[:, 1]
        o_height = self.grid["o_point"][1]
        x_bounds = [np.nanmin(x_height), np.nanmax(x_height)]
        w_height = self.w_point[1]
        if x_bounds[0] > o_height:
            x_bounds[0] = -np.inf
        if x_bounds[1] < o_height:
            x_bounds[1] = np.inf
        if w_height < x_bounds[0] or w_height > x_bounds[1]:
            return self.psi_x
        if self.polarity < 0:
            return np.min([self.psi_x, self.psi_w])
        return np.max([self.psi_x, self.psi_w])

    @property
    def psi_lcfs(self):
        """Return polodial flux at psi_norm==0.999."""
        psi_axis = self.psi_axis
        return 0.999 * (self.psi_boundary - psi_axis) + psi_axis

    @property
    def strike_points(self):
        """Return divertor strike points."""
        if self.limiter:
            return np.array([])
        levelset = self.levelset.contour.levelset(self.psi_boundary)
        self.strike.update([surface.points for surface in levelset])
        if len(strike_points := self.strike.points) == 2:
            return strike_points
        minmax_index = [np.argmin(strike_points[:, 0]), np.argmax(strike_points[:, 0])]
        return strike_points[minmax_index]

    def normalize(self, psi):
        """Return normalized flux."""
        psi_axis = self.psi_axis
        psi_boundary = self.psi_boundary
        return (psi - psi_axis) / (psi_boundary - psi_axis)

    @contextmanager
    def profile(self):
        """Update plasma current distribution."""
        try:
            psi_norm = self.normalize(self.grid.psi)
        except IndexError:
            psi_norm = None
        yield  # update separatrix
        try:  # update plasma currnet
            psi_norm = psi_norm[self.ionize]
            current_density = self.radius * self.p_prime(psi_norm) + self.ff_prime(
                psi_norm
            ) / (mu_0 * self.radius)
            current_density *= -2 * np.pi
            current = current_density * self.area
            current = abs(current)  # TODO investigate further - reverse current rejoins
            self.nturn = current / current.sum()
        except NotImplementedError:  # flux functions are not implemented
            pass

    @property
    def psi(self):
        """Manage concatenated array of grid and wall flux values."""
        return np.r_[self.grid.psi, self.wall.psi]

    @psi.setter
    def psi(self, psi):
        self.grid["psi"] = psi[: self.psi_index[0]]
        self.wall["psi"] = psi[slice(*self.psi_index[0:2])]
        with self.profile():
            self.separatrix = self.psi_lcfs

    def solve_flux(self, **kwargs):
        """Solve for equilibrium poloidal flux across plasma grid and boundary."""

        def flux_residual(psi):
            """Return flux residual."""
            self.psi = psi
            return self.psi - psi

        self.psi = newton_krylov(flux_residual, self.psi, **kwargs)

    @property
    def separatrix(self):
        """Return plasma separatrix, the convex hull of active filaments."""
        index = self.loc["plasma", "nturn"] > 0
        points = self.loc["plasma", ["x", "z"]].values[index]
        hull = scipy.spatial.ConvexHull(points)
        vertices = np.append(hull.vertices, hull.vertices[0])
        convexhull = points[vertices]
        tangent = convexhull[1:] - convexhull[:-1]
        length = np.append(0, np.cumsum(np.linalg.norm(tangent, axis=1)))
        _length = np.linspace(0, length[-1], 250)
        return np.c_[
            interp1d(length, convexhull[:, 0], "quadratic")(_length),
            interp1d(length, convexhull[:, 1], "quadratic")(_length),
        ]

    @separatrix.setter
    def separatrix(self, index):
        """
        Update plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        index : array-like (n, 2), Polygon, dict[str, list[float]], list[float]
            Bounding loop.

        """
        try:
            mask = self.grid.ionize_mask(index)
        except (AttributeError, StopIteration) as error:
            raise AttributeError(
                "use coilset.firstwall.insert " "to define plasma rejoin"
            ) from error
        self.ionize = mask
        self.nturn = self.area / np.sum(self.area)

    def plot(self, attr="psi", turns=True, axes=None, **kwargs):
        """Plot separatirx as polygon patch."""
        self.axes = axes
        if turns:
            self.subframe.polyplot("plasma")
        else:
            poly = Polygon(self.separatrix).poly
            if not poly.is_empty:
                self.axes.add_patch(
                    self.patch(
                        poly.__geo_interface__,
                        facecolor="C4",
                        alpha=0.75,
                        linewidth=0,
                        zorder=-10,
                    )
                )
        levels = self.levelset.plot(attr, **kwargs)
        if levels is None:
            levels = self.grid.plot(attr, **kwargs)
        self.wall.plot(nulls=kwargs.get("nulls", False))
        return levels
