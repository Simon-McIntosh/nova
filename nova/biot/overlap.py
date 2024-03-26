"""Solve intergral coil forces."""

from dataclasses import dataclass, field

import numpy as np
import scipy
import xarray

from nova.biot.biotframe import Target
from nova.biot.grid import Gridgen, Expand
from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.graphics.plot import Plot2D


@dataclass
class Overlap(Plot2D, Operate):
    """
    Compute error field overlap external forcing.

    Parameters
    ----------
    nloop : int, optional
        Toroidal resolution. The default is 120.

    """

    ngrid: int | None = None
    noverlap: int | None = None
    attrs: list[str] = field(default_factory=lambda: ["Br", "Bz"])

    @property
    def number(self) -> tuple[int] | None:
        """Manage poloidal and toroidal grid number."""
        if self.ngrid is None or self.noverlap is None:
            return None
        return self.ngrid, self.noverlap

    @number.setter
    def number(self, number: int | tuple[int]):
        match number:
            case (int() | float(), int() | float()):
                self.ngrid, self.noverlap = number
            case int() | float():
                self.ngrid = number

    @property
    def nphi(self):
        """Return toroidal grid number."""
        return 2 * self.noverlap

    @property
    def phi(self):
        """Return phi grid coordinate."""
        return np.linspace(0, 2 * np.pi, self.nphi)

    def _generate(self, limit, index, grid):
        """Generate simulation grid."""
        if len(grid) > 0:
            assert all([attr in grid for attr in "rz"])
            assert np.ndim(grid.r) == 1
            self.ngrid = np.prod(grid.r.shape)
            boundary = np.c_[grid.r, grid.z]
            assert np.allclose(boundary[0], boundary[-1])
            if "theta" not in grid:
                tangent = np.zeros_like(boundary)
                tangent[0] = tangent[-1] = boundary[1] - boundary[-1]
                tangent[1:-1] = boundary[2:] - boundary[:-2]
                grid.coords["theta"] = np.unwrap(
                    np.arctan2(-tangent[:, 0], tangent[:, 1])
                )
            grid.coords["phi"] = self.phi
            grid["X"] = ("theta", "phi"), grid.r.data[:, np.newaxis] * np.cos(self.phi)
            grid["Y"] = ("theta", "phi"), grid.r.data[:, np.newaxis] * np.sin(self.phi)
            grid["Z"] = ("theta", "phi"), grid.r.data[:, np.newaxis] * np.ones(
                self.nphi, float
            )
            grid.attrs["space_shape"] = (grid.sizes["theta"], self.nphi)
            grid.attrs["frequency_shape"] = (grid.sizes["theta"], self.noverlap + 1)
            return grid
        if self.number is None:
            return None
        if isinstance(limit, (int, float)):
            limit = Expand(self.subframe, index)(limit)
        match len(limit):
            case 2 | 4:  # 2d grid limits
                gridgen = Gridgen(self.ngrid, limit)
                grid = xarray.Dataset(
                    coords={
                        "r": gridgen.data.x.data,
                        "phi": self.phi,
                        "z": gridgen.data.z.data,
                    }
                )
                Radius, Phi, Height = np.meshgrid(
                    grid.r, grid.phi, grid.z, indexing="ij"
                )
                grid["R"] = ("r", "phi", "z"), Radius
                grid["Phi"] = ("r", "phi", "z"), Phi
                grid["X"] = ("r", "phi", "z"), Radius * np.cos(Phi)
                grid["Y"] = ("r", "phi", "z"), Radius * np.sin(Phi)
                grid["Z"] = ("r", "phi", "z"), Height
                grid.attrs["space_shape"] = (
                    gridgen.shape[0],
                    self.nphi,
                    gridgen.shape[1],
                )
                grid.attrs["frequency_shape"] = (
                    gridgen.shape[0],
                    self.noverlap + 1,
                    gridgen.shape[1],
                )
                return grid
            case _:
                raise NotImplementedError("overlap requires a 2d grid limit.")

    def decompose(self):
        """Decompose Boit attributes."""
        self.data.coords["mode_number"] = np.arange(0, self.noverlap + 1)
        shape = self.data.space_shape + (self.data.sizes["source"],)
        attrs = []
        for attr in self.attrs:
            variable = self.data[attr].data.reshape(shape)
            coef = scipy.fft.rfft(variable, axis=1).reshape(-1, shape[-1])
            self.data[f"{attr}_real"] = ("frequency", "source"), np.real(coef)
            self.data[f"{attr}_imag"] = ("frequency", "source"), np.imag(coef)
            attrs.extend([f"{attr}_real", f"{attr}_imag"])
        self.attrs.extend(attrs)

    def solve(
        self,
        number: int | tuple[int, int] | None = None,
        limit: float | np.ndarray | None = 0,
        index: str | slice | np.ndarray = slice(None),
        grid: xarray.Dataset = xarray.Dataset(),
    ):
        """
        Extract solve magnetic field across grid.

        Parameters
        ----------
        number : int | tuple[int, int] | None, optional
            Grid resolution [poloidal, toroidal]. The default is None which resolves to
            [Biot.ngrid, Biot.noverlap].

            - int: poloidal resolution if len(grid) == 0, else toroidal resolution
            - tuple[int, int]: poloidal and toroidal resolution

        limit : float | np.ndarray | None, optional

            float: grid expantion factor beyond coil limits. The default is 0.
            np.ndarray: Radial and vertical limits for poloidal grid.

        index : str | slice | np.ndarray, optional
            Coil index used by grid expantion factor routine when type(limit) is float.
            The default is slice(None).
        grid : xarray.Dataset, optional
            Poloidal grid. The default is xarray.Dataset().

        Returns
        -------
        None.

        """
        self.number = number
        grid = self._generate(limit, index, grid)
        with self.solve_biot(number) as number:
            if number is not None:
                target = Target(
                    {attr.lower(): grid[attr].data.ravel() for attr in "XYZ"},
                    label="Point",
                )
                self.data = Solve(
                    self.subframe, target, attrs=self.attrs, name=self.name
                ).data
                self.data = self.data.merge(
                    grid, compat="override", combine_attrs="drop_conflicts"
                )
                self.decompose()

    @property
    def shape(self):
        """Return grid shape."""
        return self.data.space_shape

    @property
    def shapes(self):
        """Extend operator shapes with frequency domain."""
        return super().shapes | {"frequency": self.data.frequency_shape}

    def plot(self, attr, mode=0, axes=None):
        """Plot error field component."""
        self.get_axes("2d", axes=axes)
        self.axes.contour(
            self.data.R[:, 0],
            self.data.Z[:, 0],
            np.log(getattr(self, f"{attr}_")[:, mode]),
        )

    '''
    def bar(self, attr: str, index=slice(None), axes=None, **kwargs):
        """Plot per-coil force component."""
        self.get_axes("1d", axes)
        if isinstance(index, str):
            index = [name in self.loc[index, :].index for name in self.coil_name]
        names = self.coil_name[index]
        self.axes.bar(names, 1e-6 * getattr(self, attr)[index], **kwargs)
        self.axes.set_xticklabels(names, rotation=90, ha="center")
        label = {"fr": "radial", "fz": "vertical"}
        self.axes.set_ylabel(f"{label[attr]} force MN")
    

    def plot(self, scale=1, norm=None, axes=None, **kwargs):
        """Plot force vectors and intergration points."""
        self.get_axes("2d", axes)
        vector = np.c_[self.fr, self.fz]
        if norm is None:
            norm = np.max(np.linalg.norm(vector, axis=1))
        length = scale * vector / norm
        patch = self.mpl["patches"].FancyArrowPatch
        if self.reduce:
            tail = np.c_[self.data.xo, self.data.zo]
        else:
            tail = np.c_[self.data.x, self.data.z]
        arrows = [
            patch(
                (x, z),
                (x + dx, z + dz),
                mutation_scale=1,
                arrowstyle="simple,head_length=0.4, head_width=0.3," " tail_width=0.1",
                shrinkA=0,
                shrinkB=0,
            )
            for x, z, dx, dz in zip(tail[:, 0], tail[:, 1], length[:, 0], length[:, 1])
        ]
        collections = self.mpl.collections.PatchCollection(
            arrows, facecolor="black", edgecolor="darkgray"
        )
        self.axes.add_collection(collections)
        return norm
    '''


if __name__ == "__main__":

    from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetric

    coilset = CoilsNonAxisymmetric(115001, 2, ngrid=500, noverlap=5)
    coilset.saloc["Ic"] = 1e3

    coilset.overlap.solve(limit=1.5)

    coilset.plot()
    coilset.overlap.plot("br_abs", 5)
