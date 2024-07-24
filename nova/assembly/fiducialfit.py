"""Manage fitting algorithums for TF coils and SSAT sectors."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import xarray

from nova.assembly.fiducialdata import FiducialData
from nova.assembly.fiducialplotter import FiducialPlotter
from nova.assembly.sectordata import SectorData
from nova.assembly.transform import Rotate


@dataclass
class FiducialFit(FiducialData):
    """Extend FiducialData class to include fitting algorithums."""

    filename: str = "fiducial_fit"
    variance: float | str = "file"
    infer: bool = True
    method: str = "rms"
    samples: int = 10
    radial_offset: float = (33.04 - 36) / (2 * np.pi)
    data: xarray.Dataset = field(init=False, repr=False, default_factory=xarray.Dataset)

    weights: ClassVar[list[float]] = [1, 1, 0.5]

    @property
    def fiducial_attrs(self):
        """Extend fiducial_attrs to incude fit parameters."""
        return super().fiducial_attrs | {
            attr: getattr(self, attr)
            for attr in ["infer", "method", "samples", "radial_offset", "weights"]
        }

    def build(self):
        """Extend build to include fiducial fitting."""
        super().build()
        self.data = self.data.rename(dict(space="cartesian"))
        self.load_target()
        self.load_measurement()
        self.evaluate_gpr("fiducial", "gpr")
        self.fit()
        self.evaluate_gpr("fiducial_fit", "fit_gpr")

    def write(self, sheet="FAT IO"):
        """Write fits to source xls files."""
        for sector in tqdm(self.data.sector.data, "updating xls workbooks"):
            sectordata = SectorData(sector)
            coils = self.data.coils.sel(sector=sector)
            with sectordata.openbook(), sectordata.savebook():
                worksheet = sectordata.book[sheet]
                for coil, xls_index in zip(coils, sectordata._coil_index(sheet)):
                    data = self.data.fiducial_fit.sel(coil=coil).sortby("target").data
                    std = (
                        self.data.fiducial_fit_gpr_std.sel(coil=coil)
                        .sortby("target")
                        .data
                    )
                    sectordata.write(
                        worksheet, xls_index, np.append(data, 2 * std, axis=1)
                    )
                    opt_x = self.data.opt_x.sel(coil=coil)
                    self._write_transform(worksheet, xls_index, opt_x)

    def _write_transform(self, worksheet, xls_index, opt_x):
        """Write transform to worksheet."""
        worksheet.cell(xls_index[0] - 4, xls_index[1] + 2, "transform")
        worksheet.cell(xls_index[0] - 5, xls_index[1] + 6, "Intrinsic Euler angles")
        for j, (label, value) in enumerate(zip(opt_x.transform.data, opt_x.data)):
            match len(label):
                case 1:
                    label = f"d{label} [mm]"
                case 2:
                    label = f"{label[0].upper()} [deg]"
            worksheet.cell(xls_index[0] - 4, xls_index[1] + 3 + j, label)
            worksheet.cell(xls_index[0] - 3, xls_index[1] + 3 + j, value)

    def load_target(self):
        """Load target geometories in cylindrical coordinate system."""
        self.data["centerline_target"] = (
            xarray.DataArray(1, [("coil", self.data.coil.data)])
            * self.data.centerline_target
        )
        for attr in ["fiducial_target", "centerline_target"]:
            self.data[f"{attr}_cyl"] = Rotate.to_cylindrical(self.data[attr])

    def load_measurement(self):
        """Load reference measurements."""
        self.data["fiducial"] = self.data.fiducial_target + self.data.fiducial_delta
        self.data["centerline"] = (
            self.data.centerline_target + self.data.centerline_delta
        )

    def evaluate_gpr(self, target="fiducial", postfix="gpr"):
        """Evaluate gpr in cylindrical coordinate system."""
        delta = Rotate.to_cylindrical(self.data[target]) - self.data.fiducial_target_cyl
        fiducial = f"fiducial_{postfix}"
        fiducial_std = f"fiducial_{postfix}_std"
        centerline = f"centerline_{postfix}"
        sample = f"sample_{postfix}"
        self.data[fiducial] = xarray.zeros_like(self.data.fiducial_target_cyl)
        self.data[fiducial_std] = xarray.zeros_like(self.data.fiducial_target_cyl)
        self.data[centerline] = xarray.zeros_like(self.data.centerline_target_cyl)
        self.data[sample] = (
            xarray.zeros_like(self.data[centerline])
            .expand_dims(dict(samples=self.samples), axis=-1)
            .copy()
        )
        for coil_index in range(self.data.sizes["coil"]):
            for space_index in range(self.data.sizes["cylindrical"]):
                self.load_gpr(coil_index, space_index)
                self.gpr.fit(delta[coil_index, :, space_index])
                (
                    self.data[fiducial][coil_index, :, space_index],
                    self.data[fiducial_std][coil_index, :, space_index],
                ) = self.gpr.predict(
                    self.data.target_length[coil_index], return_std=True
                )
                self.data[centerline][coil_index, :, space_index] = self.gpr.predict(
                    self.data.arc_length
                )
                self.data[sample][coil_index, :, space_index, :] = self.gpr.sample(
                    self.data.arc_length, self.samples
                )
        self.data[fiducial] += self.data.fiducial_target_cyl
        self.data[centerline] += self.data.centerline_target_cyl
        self.data[sample] += self.data.centerline_target_cyl
        for attr in [fiducial, centerline, sample]:
            self.data[attr] = Rotate.to_cartesian(self.data[attr])

    def plot_samples(self, label: str, coil_index: int, samples=10):
        """Load gaussian process regressor."""
        delta = Rotate.to_cylindrical(self.data[label]) - self.data.target_cyl
        axes = plt.subplots(3, 1, sharex=True)[1]
        for space_index in range(self.data.sizes["cylindrical"]):
            self.gpr.fit(delta[coil_index, :, space_index])
            self.gpr.predict(self.data.arc_length)
            self.gpr.plot(
                axes[space_index], text=False, marker="X", color="C1", line_color="C0"
            )
            if samples > 0:
                self.gpr.sample(self.data.arc_length, samples)
                self.gpr.plot_samples(axes[space_index])

            self.gpr.predict(self.data.target_length)
            axes[space_index].plot(self.data.target_length, self.gpr.data.y_mean, "dC0")
            coord = str(self.data.cylindrical[space_index].values)
            coord = coord.replace("phi", r"\phi")
            axes[space_index].set_ylabel(rf"${coord}$")
        axes[-1].set_xlabel("arc length")
        axes[0].set_title(f"TF{self.data.coil[coil_index].values:1d}")
        plt.tight_layout()
        plt.savefig("gpr.png")

    def transform(self, x, points) -> xarray.DataArray:
        """Return points transformed by vector x."""
        points = points[:] + x[:3]
        if len(x) == 6:
            rotate = Rotation.from_euler("XYZ", x[-3:], degrees=True)
            points[:] = rotate.apply(points.data)
        return points

    def delta(self, points, coil):
        """Return coil-frame deltas."""
        offset = np.zeros_like(points)
        offset[:, 0] -= self.radial_offset
        return (
            Rotate.to_cylindrical(points)
            + offset
            - Rotate.to_cylindrical(self.data.fiducial_target.loc[coil])
        )

    @staticmethod
    def error_vector(delta, method):
        """Return error vector."""
        error = np.zeros(3)
        match method:
            case "rms":
                error[0] = np.mean(delta[..., [5, 3, 4], 0] ** 2)
                error[1] = np.mean(delta[..., 1] ** 2)
                error[2] = np.mean(delta[..., [2, 1, -1, -2], 2] ** 2)
            case "max":
                error[0] = np.max(abs(delta[..., [5, 3, 4], 0]))  # radial (A, B, H)
                error[1] = np.max(abs(delta[..., 1]))  # toroidal (all)
                error[2] = np.max(abs(delta[..., [2, 1, -1, -2], 2]))  # (C, D, E, F)
            case _:
                raise NotImplementedError(f"Method {method} not implemented.")
        return error

    def transform_error(self, x, points, coil, method):
        """Return transform error vector."""
        points = self.transform(x, points)
        return self.point_error(points, coil, method)

    def weighted_transform_error(self, x, points, coil, method):
        """Return weighted transform error vector."""
        return self.transform_error(x, points, coil, method=method) * self.weights

    def point_error(self, points, coil, method=None):
        """Return error vector."""
        if method is None:
            method = self.method
        delta = self.delta(points, coil)
        return self.error_vector(delta, method)

    def max_transform_error(self, x, points, coil):
        """Return maximum error."""
        return np.max(self.weighted_transform_error(x, points, coil, method="max"))

    def rms_transform_error(self, x, points, coil):
        """Return mean error."""
        return np.sqrt(
            np.mean(self.weighted_transform_error(x, points, coil, method="rms"))
        )

    def scalar_error(self, x, points, coil):
        """Return scalar mesure for fit error."""
        return getattr(self, f"{self.method}_transform_error")(x, points, coil)

    @property
    def point_name(self):
        """Return reference point name."""
        if self.infer:
            return "fiducial_gpr"
        return "fiducial"

    def points(self, coil):
        """Return reference points."""
        return self.data[self.point_name].sel(coil=coil)

    @staticmethod
    def join(name: str, post_fix: str):
        """Return variable name with post_fix if set."""
        if post_fix:
            return "_".join([name, post_fix])
        return name

    def fit(self):
        """Perform sector fit."""
        transform_attrs = [
            "fiducial",
            "centerline",
            "fiducial_gpr",
            "centerline_gpr",
        ]
        for attr in transform_attrs:
            self.data[f"{attr}_fit"] = xarray.zeros_like(self.data[attr])

        # self.data["centerline_target_fit"] =
        self.data.coords["transform"] = ["x", "y", "z", "xx", "yy", "zz"]
        self.data["opt_x"] = xarray.DataArray(
            0.0,
            coords=[self.data.coil, self.data.transform],
            dims=["coil", "transform"],
        )
        for post_fix in ["", "gpr"]:
            error_attr = self.join("error", post_fix)
            self.data[error_attr] = xarray.DataArray(
                0.0,
                coords=[self.data.coil],
                dims=["coil"],
            )
            self.data[f"{error_attr}_fit"] = xarray.zeros_like(self.data[error_attr])
        for coil in tqdm(self.data.coil, "fitting coils"):
            points = self.points(coil=coil)
            xo = np.zeros(self.data.sizes["transform"])
            opt = minimize(self.scalar_error, xo, method="SLSQP", args=(points, coil))
            if not opt.success:
                warnings.warn(f"optimization failed {opt}")
            self.data["opt_x"].loc[{"coil": coil}] = opt.x
            for attr in transform_attrs:
                self.data[f"{attr}_fit"].loc[{"coil": coil}] = self.transform(
                    opt.x, self.data[attr].loc[{"coil": coil}].copy()
                )
            for post_fix in ["", "gpr"]:
                error_attr = self.join("error", post_fix)
                fiducial_attr = self.join("fiducial", post_fix)
                fiducial_points = self.data[fiducial_attr].sel(coil=coil)
                self.data[error_attr].loc[{"coil": coil}] = self.scalar_error(
                    xo, fiducial_points, coil
                )
                self.data[f"{error_attr}_fit"].loc[{"coil": coil}] = self.scalar_error(
                    opt.x, fiducial_points, coil
                )

    @cached_property
    def plotter(self):
        """Return FiducialPlotter instance."""
        return FiducialPlotter(self.data, factor=500)

    def plot_fit(self, coil_index, postfix=""):
        """Plot fits."""
        self.plotter.target(coil_index)
        stage = 1 + int(self.infer)
        stage = 2
        self.plotter(postfix, stage, coil_index)
        title = f"Coil{self.data.coil[coil_index].data}"
        title += f"\norigin:{self.data.origin[coil_index].data}"
        title += f" phase:{self.phase}"
        title += f"\ninfer:{self.infer} method: {self.method}"
        self.plotter.axes[0].set_title(title, fontsize="large")
        if postfix[-3:] == "fit":
            self.text_fit(self.plotter.axes[0], coil_index)
            self.text_transform(self.plotter.axes[0], coil_index)

    def plot_transform(self, coil_index=0):
        """Plot transform text."""
        self.plotter("target")
        self.text_transform(self.plotter.axes[0], coil_index)
        self.plotter.axes[0].set_title("transform: reference -> fit")

    def text_transform(self, axes, coil_index):
        """Display text transform."""
        opt_x = self.data.opt_x[coil_index].values
        deg_to_mm = 10570 * np.pi / 180
        angle_unit = "mm"  # r"$^o$"
        axes.text(
            0.3,
            0.5,
            f"dx: {opt_x[0]:1.2f}mm\n"
            + f"dy: {opt_x[1]:1.2f}mm\n"
            + f"dz: {opt_x[2]:1.2f}mm\n"
            + f"rx: {opt_x[3]*deg_to_mm:1.2}"
            + angle_unit
            + "\n"
            + f"ry: {opt_x[4]*deg_to_mm:1.2}"
            + angle_unit
            + "\n"
            + f"rz: {opt_x[5]*deg_to_mm:1.2}"
            + angle_unit,
            va="center",
            ha="left",
            transform=axes.transAxes,
            fontsize="small",
        )

    def reference_error(self, method: str):
        """Return reference error vector."""
        return self.point_error(self.data.reference.copy(), method)

    def text_fit(self, axes, coil_index):
        """Display text transform."""
        points = self.data[self.point_name][coil_index]
        opt_x = self.data.opt_x[coil_index].data
        coil = self.data.coil[coil_index].data
        self.transform_error(opt_x, points, coil, "rms")
        error = {
            "rms": self.transform_error(opt_x, points, coil, "rms"),
            "max": self.transform_error(opt_x, points, coil, "max"),
        }
        text = ""
        for i, coordinate in enumerate(
            ["radial: A,B,H", "toroidal: all", "vertical: C,D,E,F"]
        ):
            text += "\n" + coordinate + "\n"
            for method in ["rms", "max"]:
                text += f"    {method}: {error[method][i]:1.2f}\n"
        axes.text(
            0.9,
            0.5,
            text,
            va="center",
            ha="left",
            transform=axes.transAxes,
            fontsize="small",
        )

    def _get_delta(self, attr, fit=True):
        """Return ensemble deltas."""
        source_attr = attr
        if fit:
            source_attr += "_fit"
        return self.data[source_attr] - self.data[f"{attr}_target"]

    def plot_ensemble(self, fit=True, factor=250):
        """Plot fit ensemble."""
        self.axes = self.set_axes("2d", nrows=1, ncols=2, sharey=True)
        for j in range(2):
            self.axes[j].plot(
                self.data.centerline_target[0, :, 0],
                self.data.centerline_target[0, :, 2],
                "gray",
                ls="--",
            )
        limits = self.axes_limit
        color = [0, 0]

        centerline_delta = self._get_delta("centerline", fit)
        fiducial_delta = self._get_delta("fiducial", fit)

        for i in range(self.data.sizes["coil"]):
            j = 0 if self.data.origin[i] == "EU" else 1
            self.axes[j].plot(
                self.data.centerline_target[i, :, 0]
                + factor * centerline_delta[i, :, 0],
                self.data.centerline_target[i, :, 2]
                + factor * centerline_delta[i, :, 2],
                color=f"C{color[j]}",
                label=f"{self.data.coil[i].values:02d}",
            )
            self.axes[j].plot(
                self.data.fiducial_target[i, :, 0] + factor * fiducial_delta[i, :, 0],
                self.data.fiducial_target[i, :, 2] + factor * fiducial_delta[i, :, 2],
                ".",
                color=f"C{color[j]}",
            )
            color[j] += 1
        for j, origin in enumerate(["EU", "JA"]):
            self.axes[j].legend(
                fontsize="large", loc="center", bbox_to_anchor=[0.4, 0.5]
            )
            self.axes[j].set_title(f"{origin} {self.phase}")
        self.axes_limit = limits


if __name__ == "__main__":
    phase = "FAT supplier"
    # phase = "SSAT BR"

    fiducial = FiducialFit(phase=phase, infer=True, fill=False, method="rms")

    coil_index = 16
    fiducial.plot_fit(coil_index)
    fiducial.plot_fit(coil_index, "fit")

    # fiducial.plot_ensemble(True, 250)

    # fiducial.write()

    """
    for coil in range(18):
        try:
            coil_index = fiducial.data.coil.sel(coil=coil).location.data
        except (KeyError, IndexError):
            continue
        fiducial.plotter.reset_axes()
        fiducial.plot_fit(coil_index)
        fiducial.plot_fit(coil_index, "fit")
        plt.tight_layout()
        plt.savefig(f"IDM_TF{coil}_fit.png")
    """

    # fiducial.plot_transform()

    # fiducial.plot_fit("target")
    # print(fiducial.data.target_cyl)

    # fiducial.plot()

    # coil = 4
    # coil_index = fiducial.coil_index(coil)
