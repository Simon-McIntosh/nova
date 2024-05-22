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
    def number(self) -> tuple[int, int] | None:
        """Manage poloidal and toroidal grid number."""
        if self.ngrid is None or self.noverlap is None:
            return None
        return self.ngrid, self.noverlap

    @number.setter
    def number(self, number: int | tuple[int, int] | None):
        match number:
            case (int() | float(), int() | float()):
                self.ngrid, self.noverlap = number
            case int() | float():
                self.ngrid = number

    @property
    def nphi(self):
        """Return toroidal grid number."""
        return 2 * self.noverlap + 1

    @property
    def phi(self):
        """Return phi grid coordinate."""
        return np.linspace(0, 2 * np.pi, self.nphi, endpoint=True)

    def _generate(self, limit, index, grid):
        """Generate simulation grid."""
        if len(grid) > 0:
            assert all([attr in grid for attr in "rz"])
            assert np.ndim(grid.r) == 1
            self.ngrid = np.prod(grid.r.shape)
            boundary = np.c_[grid.r, grid.z]
            tangent = np.zeros_like(boundary)
            tangent[1:-1] = boundary[2:] - boundary[:-2]
            if np.allclose(boundary[0], boundary[-1]):
                tangent[0] = tangent[-1] = boundary[1] - boundary[-1]
            else:
                tangent[0] = boundary[1] - boundary[0]
                tangent[-1] = boundary[-1] - boundary[-2]
            tangent = np.insert(tangent, 1, np.zeros(len(tangent)), axis=1)
            normal = np.cross(tangent, np.array([[0, 1, 0]]))
            grid.coords["axes"] = list("rz")
            grid["normal"] = ("theta", "axes"), normal[:, ::2]
            grid["normal"] /= np.linalg.norm(grid["normal"], axis=1)[:, np.newaxis]
            grid.coords["phi"] = self.phi
            grid["X"] = ("theta", "phi"), grid.r.data[:, np.newaxis] * np.cos(self.phi)
            grid["Y"] = ("theta", "phi"), grid.r.data[:, np.newaxis] * np.sin(self.phi)
            grid["Z"] = ("theta", "phi"), grid.z.data[:, np.newaxis] * np.ones_like(
                self.phi
            )
            grid.attrs["shape_space"] = (grid.sizes["theta"], self.nphi)
            grid.attrs["shape_n"] = (grid.sizes["theta"], self.noverlap + 1)
            grid.attrs["shape_mn"] = (grid.sizes["theta"] - 1, self.noverlap + 1)
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
                grid.attrs["shape_space"] = (
                    gridgen.shape[0],
                    self.nphi,
                    gridgen.shape[1],
                )
                grid.attrs["shape_n"] = (
                    gridgen.shape[0],
                    self.noverlap + 1,
                    gridgen.shape[1],
                )
                grid.attrs["shape_mn"] = ()
                return grid
            case _:
                raise NotImplementedError("overlap requires a 2d grid limit.")

    def compose(self):
        """Assemble overlap attributes."""
        self.attrs = [
            attr for attr in self.attrs if attr.split("_")[-1] not in ["real", "imag"]
        ]

    def decompose(self):
        """Decompose Boit attributes."""
        self.data.coords["mode_number"] = np.arange(0, self.noverlap + 1)
        shape = self.data.shape_space + (self.data.sizes["source"],)
        if np.all([attr in self.data for attr in ["Br", "Bz", "theta"]]):
            self.data["Bn"] = ("target", "source"), np.einsum(
                "ij,ij...->i...",
                self.data.normal,
                np.stack(
                    [
                        self.data["Br"].data.reshape(shape),
                        self.data["Bz"].data.reshape(shape),
                    ],
                    axis=1,
                ),
            ).reshape((self.data.sizes["target"], self.data.sizes["source"]))
            self.attrs.append("Bn")
        attrs = []
        for attr in self.attrs:
            variable = self.data[attr].data.reshape(shape)
            coef_n = scipy.fft.rfft(variable[:, :-1], norm="forward", axis=1)

            self.data[f"{attr}_real"] = ("target_n", "source"), np.real(
                coef_n.reshape(-1, shape[-1])
            )
            self.data[f"{attr}_imag"] = ("target_n", "source"), np.imag(
                coef_n.reshape(-1, shape[-1])
            )
            attrs.extend([f"{attr}_real", f"{attr}_imag"])

            if attr == "Bn":
                coef_mn = scipy.fft.fftshift(
                    scipy.fft.fft(coef_n[:-1], norm="forward", axis=0), axes=0
                )
                self.data["Bmn_real"] = ("target_mn", "source"), np.real(
                    coef_mn.reshape(-1, shape[-1])
                )
                self.data["Bmn_imag"] = ("target_mn", "source"), np.imag(
                    coef_mn.reshape(-1, shape[-1])
                )
                attrs.extend(["Bmn_real", "Bmn_imag"])

        self.attrs.extend(attrs)

    def solve(
        self,
        number: int | tuple[int] | None = None,
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
                self.compose()
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
        return self.data.shape_space

    @property
    def shapes(self):
        """Extend operator shapes with frequency domain."""
        return super().shapes | {
            "target_n": self.data.shape_n,
            "target_mn": self.data.shape_mn,
        }

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

    coilset = CoilsNonAxisymmetric(115001, 2, ngrid=500, noverlap=80)
    coilset.saloc["Ic"] = 1e3

    coilset.overlap.solve(limit=1.5)

    coilset.plot()
    coilset.overlap.plot("br_abs", 5)
