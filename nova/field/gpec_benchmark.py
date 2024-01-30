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
        115001, 2, minimum_arc_nodes=1000, field_attrs=["Bx", "By", "Bz"]
    )

    # elm_ids.grid.solve(2e3, limit=dataset.grid.limit)
    elm_ids.sloc["Ic"] = 0
    elm_ids.sloc["Ic"][0] = 1

    elm_ids.grid.plot("bz", levels=51, nulls=False)
    elm_ids.plot(axes=elm_ids.grid.axes)
