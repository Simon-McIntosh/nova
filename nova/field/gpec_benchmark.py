"""Benchmark error field calculation against GPEC code."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas
import xarray

from nova.biot.grid import Gridgen
from nova.database.filepath import FilePath
from nova.database.netcdf import netCDF
from nova.graphics.plot import Plot2D


@dataclass
class Dataset(Plot2D, netCDF):
    """Locate GPEC dataset."""

    filename: str
    datadir: str = "/mnt/share/gpec_benchmark"

    def __post_init__(self):
        """Load / build gpec dataset."""
        super().__post_init__()
        try:
            self.load()
        except FileNotFoundError:
            self.build()
            self.store()

    @cached_property
    def gpec_filename(self):
        """Return gpec filename."""
        match list(self.coil):
            case "E", str(position):
                gpec_position = {"E": "M"}.get(position, position)
                gpec_coil = f"RMP_{gpec_position}"
            case _:
                raise NotImplementedError(
                    f"GPEC filename not implemented for coil {self.coil}"
                )
        return f"{gpec_coil}_cbrzphi_n{self.mode_number}.out"

    @property
    def coil(self):
        """Return filepath attribute."""
        return self.filename

    @cached_property
    def mode_number(self) -> int:
        """Return toroidal mode number."""
        match list(self.coil):
            case "E", str():
                return 3
            case str(), "C", "C":
                return 1
            case _:
                raise NotImplementedError(
                    "mode number for coil {self.coil} not implemented"
                )

    def build(self):
        """Read dataset from .out file."""
        self.data = xarray.Dataset()
        datafile = FilePath(self.gpec_filename, self.datadir)
        assert datafile.is_file()
        with open(datafile.filepath, "r") as file:
            self.data.attrs["name"] = file.readline().strip()
            self.data.attrs["version"] = file.readline().strip()
            current = file.readline().split(":")[1].split(",")
            self.data["current"] = "coil", np.array(current, dtype=float)
            self.data.attrs["mode"] = int(file.readline().split("=")[1])
            grid = file.readline().split()
            self.data.attrs["nr"] = int(grid[2])
            self.data.attrs["nz"] = int(grid[5])
            dataframe = pandas.read_csv(
                file,
                skiprows=1,
                delim_whitespace=True,
            )
        self.data["r2d"] = ("r", "z"), dataframe.r.values.reshape(
            self.data.nr, self.data.nz
        )
        self.data.coords["r"] = self.data["r2d"].data[:, 0]
        self.data["z2d"] = ("r", "z"), dataframe.z.values.reshape(
            self.data.nr, self.data.nz
        )
        self.data.coords["z"] = self.data["z2d"].data[0, :]
        self.data.coords["complex"] = ["real", "imag"]
        for attr in ["b_r", "b_phi", "b_z"]:
            self.data[attr.replace("_", "")] = ("r", "z", "complex"), np.stack(
                [dataframe[f"real({attr})"].values, dataframe[f"imag({attr})"].values],
                axis=-1,
            ).reshape(self.data.nr, self.data.nz, 2)

    def plot(self, attr="br", component="amplitude"):
        """Plot GPEC attribute."""
        match component:
            case "amplitude":
                data = np.linalg.norm(self.data[attr], axis=-1)
                self.axes.contour(self.data.r2d, self.data.z2d, np.log(data))

    @cached_property
    def grid(self):
        """Return gridgen instance."""
        return Gridgen(limit=np.stack([self.data.r, self.data.z]))


if __name__ == "__main__":
    dataset = Dataset("EU")

    # dataset.plot()

    # dataset.grid.plot()

    from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetyric

    elm_ids = CoilsNonAxisymmetyric(
        115001, 2, field_attrs=["Bx", "By", "Bz", "Ax", "Ay", "Az"]
    )

    resolve = False
    if resolve:
        elm_ids.grid.solve(5e3, [2.5, 6, -4, 4])  # limit=dataset.grid.limit
        elm_ids._clear()
        elm_ids.store()

    from nova.frame.coilset import CoilSet

    elm_ids = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay", "Az", "Psi"])
    elm_ids.coil.insert(8, 0, 0.5, 0.5)
    elm_ids.grid.solve(5e3, [2.5, 12, -4, 4])

    elm_ids.sloc["Ic"] = 1e3
    # elm_ids.sloc["Ic"][8] = 1
    # elm_ids.sloc["Ic"][-1] = 1

    # elm_ids.grid.plot("bx", levels=31, nulls=False)
    elm_ids.plot(axes=elm_ids.grid.axes)
    # elm_ids.frame.polyplot(axes=elm_ids.grid.axes, index=[8])

    grid = elm_ids.grid
    grid.axes.streamplot(grid.data.x.data, grid.data.z.data, grid.bx_.T, grid.bz_.T)

    """
    import scipy

    psi = scipy.integrate.cumulative_simpson(grid.bx_, x=grid.data.x, axis=0, initial=0)
    psi = scipy.integrate.cumulative_simpson(
        psi + grid.bz_, x=grid.data.z, axis=1, initial=0
    )
    """
    import scipy

    A = np.stack([grid.ax_, grid.ay_, grid.az_], axis=-1)
    B = np.stack([grid.bx_, grid.by_, grid.bz_], axis=-1)

    def grad(u):
        rbf = scipy.interpolate.RectBivariateSpline(grid.data.x, grid.data.z, u)
        ux = rbf.ev(grid.data.x2d, grid.data.z2d, dx=1)
        uz = rbf.ev(grid.data.x2d, grid.data.z2d, dy=1)
        return np.stack([ux, np.zeros_like(ux), uz], axis=-1)

    def res(alpha):
        dalpha = grad(alpha)

        return np.linalg.norm(np.cross(dalpha, A / alpha[..., np.newaxis]) - B, axis=-1)

    _u = grid.ay_
    alpha = scipy.optimize.newton_krylov(res, _u, iter=20, verbose=True)

    print(np.linalg.norm(res(alpha)))

    dalpha = grad(alpha)
    dbeta = A / alpha[..., np.newaxis]

    assert np.allclose(np.cross(dalpha, dbeta), B, atol=1e-3)
    assert np.allclose(alpha[..., np.newaxis] * dbeta, A, atol=1e-3)
    assert np.allclose(np.einsum("ijk,ijk->ij", dbeta, A), 0, atol=1e-3)

    grid.axes.contour(grid.data.x.data, grid.data.z.data, alpha.T, levels=51)

    grid.plot("psi")

    """
    import pyvista as pv
    import vedo

    points = np.stack(
        [
            elm_ids.grid.data.x2d,
            np.zeros_like(elm_ids.grid.data.x2d),
            elm_ids.grid.data.z2d,
        ],
        axis=-1,
    ).reshape(-1, 3)

    mesh = pv.PolyData(points).delaunay_2d()
    contours = mesh.contour(isosurfaces=71, scalars=elm_ids.grid.bx.reshape(-1))

    elm_ids.frame.vtkplot(index=["EU9B", "EE9B", "EL9B"])
    vedo.Mesh(contours, c="black").show(new=False)
    """
