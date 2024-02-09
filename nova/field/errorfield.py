"""Manage error field database."""

from dataclasses import dataclass, field
from typing import ClassVar

import imas
import json
import numpy as np
import pandas
import scipy

from nova.database.datafile import Datafile
from nova.database.filepath import FilePath
from nova.graphics.plot import Plot
from nova.imas.database import Database
from nova.imas.metadata import Metadata


@dataclass
class ErrorField(Plot, Datafile):
    """Interpolate field dataset to first wall and decompose."""

    filename: str
    surface: str | None = None
    dirname: str = ".error_field"
    datadir: str = "/mnt/data/error_field"
    datafile: str = field(init=False, default="")

    library: ClassVar[dict[str, str]] = {
        "86T4WW": "Database of magnetic field produced by Ferromagnetic "
        "Inserts magnetized by TF, CS and PF coils in 5MA/1.8T "
        "H-mode scenario",
        "88LDE3": "Database of magnetic field produced by Ferromagnetic "
        "Inserts and Test Blanket Modules magnetized by TF, CS and "
        "PF coils in 15MA reference scenario",
    }

    def __post_init__(self):
        """Set surface and filenanes."""
        self.datafile = self.filename
        if self.surface is None:
            self.surface = f"surface_{self.filename}"
        else:
            self.filename = f"{self.datafile}_{self.surface}"
        super().__post_init__()

    def _reshape(self, vector, shape):
        """Return vector reshaped as fortran array with axes 1, 2 swapped."""
        return vector.values.reshape(
            (shape[0],) + shape[-2:][::-1], order="F"
        ).swapaxes(2, 1)

    def build(self):
        """Build database from source datafile."""
        self.read_datafile()
        self.build_surface()
        self.compose()
        self.decompose()
        self.store()

    def read_datafile(self):
        """Read source datafile."""
        datafile = FilePath(dirname=self.datadir, filename=f"{self.datafile}.txt")
        with open(datafile.filepath, "r") as file:
            header = file.readline()
            data = pandas.read_csv(
                file,
                header=None,
                delim_whitespace=True,
                names=["r", "phi", "z", "Br", "Bphi", "Bz"],
            )
        shape = tuple(int(dim) for dim in header.split()[:3])
        self.data.attrs["uid"] = self.datafile
        self.data.attrs["title"] = self.library[self.datafile]
        self.data.attrs["Io"] = float(header.split()[-1])
        self.data.coords["radius"] = self._reshape(data.r, shape)[:, 0, 0]
        self.data.coords["phi"] = self._reshape(data.phi, shape)[0, :, 0]
        self.data.coords["height"] = self._reshape(data.z, shape)[0, 0, :]
        for attr in ["Br", "Bphi", "Bz"]:
            self.data[f"grid_{attr}"] = ("radius", "phi", "height"), self._reshape(
                data[attr], shape
            )

    def build_surface(self):
        """Build control surface."""
        datafile = FilePath(dirname=self.datadir, filename=f"{self.surface}.txt")
        with open(datafile.filepath, "r") as file:
            data = pandas.read_csv(
                file, header=None, delim_whitespace=True, names=["radius", "height"]
            )
        delta = np.zeros((3, len(data)))
        delta[0] = np.roll(data.radius, -1) - np.roll(data.radius, 1)
        delta[2] = np.roll(data.height, -1) - np.roll(data.height, 1)
        normal = np.cross(np.array([0, 1, 0])[np.newaxis, :], delta, axisb=0)
        normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
        self.data.coords["index"] = np.arange(len(data))
        self.data.coords["coordinate"] = ["radius", "height"]
        self.data.coords["surface"] = ("index", "coordinate"), np.c_[
            data.radius, data.height
        ]
        self.data.coords["normal"] = ("index", "coordinate"), np.c_[
            normal[:, 0], normal[:, 2]
        ]

    def compose(self):
        """Interpolate field components to control surface."""
        for attr in ["Br", "Bphi", "Bz"]:
            self.data[attr] = ("index", "phi"), self.data[f"grid_{attr}"].interp(
                dict(radius=self.data.surface[:, 0], height=self.data.surface[:, 1])
            ).data
        field = np.stack([self.data.Br, self.data.Bz], axis=-1)
        self.data["Bn"] = ("index", "phi"), np.einsum(
            "ik,ijk->ij", self.data.normal, field
        )

    def decompose(self):
        """Perform Fourier decomposition."""
        coef = scipy.fft.rfft(self.data.Bn.data)
        self.data.coords["mode_number"] = np.arange(0, coef.shape[-1])
        self.data["Bn_real"] = ("index", "mode_number"), np.real(coef)
        self.data["Bn_imag"] = ("index", "mode_number"), np.imag(coef)

    def _plot_normal(self, axes, mode=1, scale=1, skip=5):
        """Plot surface and unit normals scaled to mode amplitude."""
        axes.plot(self.data.surface[:, 0], self.data.surface[:, 1], "C0-")
        patch = self.mpl["patches"].FancyArrowPatch
        tail = self.data.surface[::skip]
        amplitude = np.sqrt(
            self.data.Bn_real[:, mode] ** 2 + self.data.Bn_imag[:, mode] ** 2
        )
        length = self.data.normal[::skip] * scale * amplitude[::skip]

        arrows = [
            patch(
                (x, z),
                (x + dx, z + dz),
                mutation_scale=0.5,
                arrowstyle="simple,head_length=0.4, head_width=0.3," " tail_width=0.1",
                shrinkA=0,
                shrinkB=0,
            )
            for x, z, dx, dz in zip(tail[:, 0], tail[:, 1], length[:, 0], length[:, 1])
        ]
        collections = self.mpl.collections.PatchCollection(
            arrows, facecolor="black", edgecolor="darkgray"
        )
        axes.text(
            np.mean(self.data.surface[:, 0]),
            np.mean(self.data.surface[:, 1]),
            f"n={mode}",
            fontsize="x-large",
            ha="center",
            va="center",
        )
        axes.add_collection(collections)
        axes.autoscale_view()

    def plot_normal(self, modes=[18], scale=1):
        """Plot Bn mode amplitudes."""
        self.set_axes("2d", ncols=len(modes))
        for i, mode in enumerate(modes):
            try:
                axes = self.axes[i]
            except TypeError:
                axes = self.axes
            self._plot_normal(axes, mode, scale)

    def plot_trace(self, index=250):
        """Plot poloidal trace."""
        coef = self.data.Bn_real + self.data.Bn_imag * 1j
        # coef[:, 20:] = 0
        ifft = scipy.fft.irfft(coef.data)

        self.set_axes("1d")
        self.axes.plot(self.data.phi, self.data.Bn[index], "C0-")
        self.axes.plot(self.data.phi, ifft[index], "C1--")
        self.axes.set_xlabel(r"$\phi$ deg")
        self.axes.set_ylabel(r"$B_n$")

    def write(self):
        """Write subset of dataset to file."""
        data = self.data[
            ["surface", "normal", "Br", "Bphi", "Bz", "Bn", "Bn_real", "Bn_imag"]
        ]
        filepath = FilePath(
            dirname=self.dirname, filename=f"external_field_{self.filename}.nc"
        ).filepath
        data.to_netcdf(filepath)

    def write_ids(self, pulse=160400, run=1):
        """Write data to ids."""
        ids = imas.b_field_non_axisymmetric()
        metadata = Metadata(ids)
        metadata.put_properties(
            self.library[self.datafile], self.datafile, homogeneous_time=1
        )
        metadata.put_code(
            "Toroidal Fourier decomposition of "
            "b-field normal to a given control surface"
        )

        ids.control_surface_names = [self.surface]

        ids.time.resize(1)
        ids.time = np.array([0])
        ids.time_slice.resize(1)
        time_slice = ids.time_slice[0]
        time_slice.field_map.name = self.library[self.datafile]
        time_slice.field_map.grid.r = self.data.radius.data
        time_slice.field_map.grid.phi = self.data.phi.data * np.pi / 180
        time_slice.field_map.grid.z = self.data.height.data
        time_slice.field_map.b_field_r = self.data.grid_Br.data
        time_slice.field_map.b_field_phi = self.data.grid_Bphi.data
        time_slice.field_map.b_field_z = self.data.grid_Bz.data

        time_slice.control_surface.resize(1)
        control_surface = time_slice.control_surface[0]
        control_surface.outline.r = self.data.surface[:, 0].data
        control_surface.outline.z = self.data.surface[:, 1].data
        control_surface.normal_vector.r = self.data.normal[:, 0].data
        control_surface.normal_vector.z = self.data.normal[:, 1].data
        control_surface.phi = self.data.phi.data * np.pi / 180
        control_surface.n_tor = self.data.mode_number.data
        control_surface.b_field_r = self.data.Br.data
        control_surface.b_field_phi = self.data.Bphi.data
        control_surface.b_field_z = self.data.Bz.data
        control_surface.b_field_normal_fourier = (
            self.data.Bn_real.data + 1j * self.data.Bn_imag.data
        )

        Database(pulse, run).put_ids(ids)

    def grid_schema(self):
        """Print schema for grid data."""
        data = self.data[["grid_Br", "grid_Bphi", "grid_Bz"]]
        data = data.rename(
            {
                attr: attr.replace("grid_", "")
                for attr in ["grid_Br", "grid_Bphi", "grid_Bz"]
            }
        )
        print(data)
        print(json.dumps(data.to_dict(False), indent=4))


if __name__ == "__main__":
    errorfield = ErrorField("88LDE3", surface="surf_gpec_131025")

    errorfield.write_ids()

    # errorfield.plot_normal(modes=[1, 2, 3], scale=20)
    # errorfield.plot_normal(modes=[18], scale=1)
    # errorfield.write()
    # errorfield.plot_trace(0)

    # errorfield.grid_schema()

    field = Database(pulse=160400, run=1, name="b_field_non_axisymmetric")
