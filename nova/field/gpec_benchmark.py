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
    group: str = "surface"
    dirname: str = ".gpec"
    datadir: str = "/mnt/share/gpec_benchmark"

    def __post_init__(self):
        """Load / build gpec dataset."""
        super().__post_init__()
        try:
            self.load()
        except (FileNotFoundError, OSError):
            self.build()
            self.store()

    @cached_property
    def gpec_filename(self):
        """Return gpec filename."""
        return f"{self.coil}_{self.group}.out"

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

    def _read_header(self, file):
        """Read header from file handle."""
        self.data.attrs["name"] = file.readline().strip()
        self.data.attrs["version"] = file.readline().strip()
        current = file.readline().split(":")[1].split(",")
        self.data["current"] = "coil", np.array(current, dtype=float)
        self.data.attrs["mode"] = int(file.readline().split("=")[1])
        if self.group == "grid":
            grid = file.readline().split()
            self.data.attrs["nr"] = int(grid[2])
            self.data.attrs["nz"] = int(grid[5])

    def _store_grid(self, dataframe):
        """Store dataframe with grid data format."""
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

    def _store_surface(self, dataframe):
        """Store dataframe with a control surface data format."""
        self.data.coords["theta"] = 2 * np.pi * dataframe["theta/2/pi"]
        self.data["r"] = "theta", dataframe["R"]
        self.data["z"] = "theta", dataframe["Z"]
        self.data["bn_real"] = "theta", dataframe.iloc[:, -2]
        self.data["bn_imag"] = "theta", dataframe.iloc[:, -1]

    def build(self):
        """Read dataset from .out file."""
        self.data = xarray.Dataset()
        datafile = FilePath(self.gpec_filename, self.datadir)
        assert datafile.is_file()
        with open(datafile.filepath, "r") as file:
            self._read_header(file)
            dataframe = pandas.read_csv(
                file,
                skiprows=1,
                delim_whitespace=True,
            )
            getattr(self, f"_store_{self.group}")(dataframe)

    def plot(self, attr="br", component="amplitude"):
        """Plot GPEC attribute."""
        match component:
            case "amplitude":
                data = np.linalg.norm(self.data[attr], axis=-1)
                self.axes.contour(self.data.r2d, self.data.z2d, np.log(data))

    @cached_property
    def grid(self):
        """Return gridgen instance."""
        if self.group == "grid":
            return Gridgen(limit=np.stack([self.data.r, self.data.z]))
        return xarray.Dataset({"r": self.data.r, "z": self.data.z})


if __name__ == "__main__":

    from tqdm import tqdm

    from nova.frame.coilset import CoilSet
    from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetric

    dataset = Dataset("EU")
    dataset.grid = dataset.grid.isel(theta=range(0, 21))

    datasource = {
        # "CC": (111003, 3),
        "ELM": (115001, 2),
    }

    coilset = CoilSet(filename="gpec_benchmark", noverlap=3)

    try:
        coilset.load()
    except FileNotFoundError:
        for coil, pulse_run in tqdm(datasource.items()):
            coilset += CoilsNonAxisymmetric(*pulse_run)
        coilset.overlap.solve(grid=dataset.grid)
        coilset.store()

    coilset.sloc["Ic"][:9] = dataset.data.current / 6
    # coilset.sloc["Ic"] = 0
    # coilset.sloc["Ic"][1] = 1e3
    mode = 2

    coilset.overlap.set_axes("1d")
    coilset.overlap.axes.plot(
        coilset.overlap.data.theta, coilset.overlap.bn_abs_[:, mode], label="NOVA"
    )
    bn_abs = np.linalg.norm([dataset.data.bn_real, dataset.data.bn_imag], axis=0)
    bn_abs[1:] /= 2

    coilset.overlap.axes.plot(
        dataset.data.theta,
        bn_abs,
        label="GPEC",
    )
    coilset.overlap.legend()
    coilset.overlap.axes.set_xlabel(r"$\theta$")
    coilset.overlap.axes.set_ylabel(r"$|C_n|$")

    coilset.overlap.set_axes("2d")
    coilset.frame.polyplot(slice(0, 9))
    coilset.overlap.axes.plot(coilset.overlap.data.r, coilset.overlap.data.z)

    coilset.overlap.set_axes("1d")
    coilset.overlap.axes.bar(range(20), coilset.overlap.bmn_abs_[:, mode])
    """
    import pyvista
    import vedo

    grid = pyvista.StructuredGrid(
        coilset.overlap.data.X.data.swapaxes(0, 1),
        coilset.overlap.data.Y.data.swapaxes(0, 1),
        coilset.overlap.data.Z.data.swapaxes(0, 1),
    )

    contours = grid.contour(isosurfaces=75, scalars=coilset.overlap.bn)

    plt = vedo.Plotter(axes=0)
    plt.add(vedo.Mesh(contours).c("white"))
    plt.add(vedo.Mesh(grid).c("gray").alpha(1))
    coilset.frame.vtkplot(new=True, interactive=False, plotter=plt)
    plt.show()
    """
