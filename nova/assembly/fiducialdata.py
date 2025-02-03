"""Load ccl fiducial data for ITER TF coilset."""

from functools import cached_property
from dataclasses import dataclass, field
from pathlib import Path
import string
from typing import ClassVar

import numpy as np
import pandas
import pyvista as pv
import xarray

from nova.assembly.centerline import CenterLine
from nova.assembly.fiducialccl import Fiducial, FiducialIDM, FiducialRE
from nova.assembly.fiducialilis import FiducialIlis
from nova.assembly.fiducialsector import FiducialSector
from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor
from nova.assembly.plotter import Plotter
from nova.assembly.sectordata import SectorData
from nova.database.netcdf import netCDF
from nova.graphics.plot import Plot


@dataclass
class FiducialData(netCDF, Plot, Plotter):
    """Manage ccl fiducial data."""

    filename: str = "fiducial_data"
    dirname: Path | str = ".nova/sector_modules"
    fiducial: Fiducial | str = "Sector"
    phase: str = "SSAT BR"
    sectors: dict[int, list] = field(
        init=True, repr=False, default_factory=lambda: dict.fromkeys(range(1, 10), [])
    )
    fill: bool = True
    variance: float | str = 0.09
    ilis: bool = True
    ilis_pcr: bool = True
    sead: int = 2030
    data: xarray.Dataset = field(init=False, repr=False, default_factory=xarray.Dataset)
    gpr: GaussianProcessRegressor = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    location: ClassVar[list[int]] = [
        14,
        15,
        4,
        17,
        6,
        7,
        2,
        3,
        16,
        5,
        12,
        13,
        8,
        9,
        10,
        11,
        18,
        1,
        19,
    ]

    def __post_init__(self):
        """Load data."""
        super().__post_init__()
        self.group = self.hash_attrs(self.fiducial_attrs)
        self.load_build()

    @property
    def fiducial_attrs(self):
        """Return fiducial attributes."""
        return {
            attr: getattr(self, attr)
            for attr in ["fiducial", "phase", "fill", "variance", "ilis", "sead"]
        } | {"coils": [coil for coils in self.sectors.values() for coil in coils]}

    def load_build(self):
        """Load or build dataset."""
        try:
            self.load()
        except (FileNotFoundError, OSError):
            self.build()

    @cached_property
    def dataset(self):
        """Return source fiducial dataset."""
        return {
            "RE": FiducialRE,
            "IDM": FiducialIDM,
            "Sector": FiducialSector,
        }[self.fiducial](self.data.target, phase=self.phase, sectors=self.sectors)

    def build(self):
        """Build fiducial dataset."""
        self.build_dataset()
        if self.fill:
            self.backfill()
        self.locate_coils()
        self.build_mesh()
        self.store()

    def store(self):
        """Extend store method to include attribute hash"""
        attrs = self.fiducial_attrs
        for attr, value in attrs.items():
            if isinstance(value, bool):
                attrs[attr] = int(value)
        self.data.attrs |= attrs
        super().store()

    def build_dataset(self):
        """Build xarray dataset."""
        self.initialize_data()
        self.load_centerline()
        self.load_fiducials()
        self.load_fiducial_deltas()

    def label_coils(self, plotter, location="OD"):
        """Add coil labels."""
        plotter.add_point_labels(
            self.mesh[location][:18], self.mesh["label"][:18], font_size=20
        )

    def backfill(self):
        """Insert samples drawn from EU/JA datasets as proxy for missing."""
        metadata = xarray.Dataset(coords=dict(DA=["EU", "JA"], coil=range(1, 20)))
        metadata["origin"] = (
            "coil",
            [
                "EU",
                "JA",
                "EU",
                "EU",
                "EU",
                "EU",
                "JA",
                "JA",
                "JA",
                "JA",
                "EU",
                "JA",
                "JA",
                "EU",
                "JA",
                "JA",
                "EU",
                "EU",
                "JA",
            ],
        )
        rng = np.random.default_rng(self.sead)  # sead random number generator
        self.data = self.data.assign_coords(
            clone=("coil", np.full(self.data.sizes["coil"], -1))
        )
        fill = []
        for DA in metadata.DA:
            source = self.data.coil[self.data.origin == DA].values
            index = metadata.coil[metadata.origin == DA].values
            target = index[~np.isin(index, source)]
            sample = rng.integers(len(source), size=len(target))
            copy = self.data.sel(coil=source[sample])
            copy = copy.assign_coords(coil=target)
            copy = copy.assign_coords(clone=("coil", source[sample]))
            fill.append(copy)
        self.data = xarray.concat([self.data, *fill], dim="coil", data_vars="minimal")
        self.data = self.data.sortby("coil")

    def locate_coils(self):
        """Update data with coil's position index."""
        self.data = self.data.assign_coords(
            location=("coil", [self.location.index(coil) for coil in self.data.coil])
        )
        self.data = self.data.sortby("location")

    def build_mesh(self):
        """Build vtk mesh."""
        self.mesh = pv.PolyData()
        for coil_index, loc in enumerate(self.data.location):
            if loc.coil == 19:
                continue
            centerline = pv.Spline(1e-3 * self.data.centerline_target[coil_index])
            centerline["arc_length"] /= centerline["arc_length"][-1]
            coil = centerline.copy()
            coil["delta"] = 1e-3 * self.data.centerline_delta.sel(coil=loc.coil)
            coil.rotate_z(
                20 * loc.values,
                point=(0, 0, 0),
                transform_all_input_vectors=True,
                inplace=True,
            )
            midplane = coil.slice(normal="z", origin=(0, 0, 0))
            midplane.points += midplane["delta"]
            coil["coil"] = [loc.coil.values]
            coil["ID"] = [midplane.points[0]]
            coil["OD"] = [midplane.points[1]]
            label = f"{loc.coil.values:02d}"
            try:
                if (clone := self.data.clone.sel(coil=loc.coil)) != -1:
                    label += f"<{clone.values}"
            except AttributeError:
                pass
            coil["label"] = [label]
            self.mesh = self.mesh.merge(coil, merge_points=False)

    def initialize_data(self):
        """Init xarray dataset."""
        self.data = xarray.Dataset(
            coords=dict(space=["x", "y", "z"], target=list(string.ascii_uppercase[:8]))
        )

    def load_fiducials(self):
        """Load ccl fiducials."""
        self.data["fiducial_target"] = (
            ("target", "space"),
            self.fiducials().loc[self.data.target, :],
        )

        target_index = [
            np.argmin(
                np.linalg.norm(self.data.centerline_target[:-1] - fiducial, axis=1)
            )
            for fiducial in self.data.fiducial_target
        ]

        self.data = self.data.assign_coords(target_index=("target", target_index))
        target_length = self.data.arc_length[target_index].values
        self.data = self.data.assign_coords(target_length=("target", target_length))
        self.data = self.data.sortby("target_length")

        delta, origin = self.dataset.data
        self.data["coil"] = list(delta)
        self.data = self.data.assign_coords(origin=("coil", origin))

        for attr in [
            "fiducial_target",
            "centerline_target",
            "target_length",
            "target_index",
        ]:
            self.data[attr] = (
                xarray.DataArray(1, [("coil", self.data.coil.data)]) * self.data[attr]
            )

        if hasattr(self.dataset, "fiducial"):
            for coil in self.data.coil.data:
                self.data["fiducial_target"].loc[coil] = self.dataset.fiducial[
                    coil
                ].loc[self.data.target, :]

                self.data["target_index"].loc[coil] = [
                    np.argmin(
                        np.linalg.norm(
                            self.data.centerline_target.loc[coil][:-1] - fiducial,
                            axis=1,
                        )
                    )
                    for fiducial in self.data["fiducial_target"].loc[coil]
                ]
                self.data["target_length"].loc[coil] = self.data.arc_length[
                    self.data["target_index"].loc[coil]
                ].values

    def load_centerline(self):
        """Load geodesic centerline."""
        centerline = CenterLine()
        self.data["arc_length"] = centerline.mesh["arc_length"]
        self.data["centerline_target"] = (
            ("arc_length", "space"),
            1e3 * centerline.mesh.points,
        )

    def load_fiducial_deltas(self):
        """Load fiducial deltas."""
        delta = self.dataset.data[0]
        self.data["fiducial_delta"] = (
            ("coil", "target", "space"),
            np.stack([delta[index].to_numpy(float) for index in delta], axis=0),
        )
        if hasattr(self.dataset, "ilis") and self.ilis:
            # adjust nose fiducials to mean ilis plane
            target = ["B", "H", "A"]
            ilis = FiducialIlis(self.dataset.ilis, pcr=self.ilis_pcr)
            # map fiducial deltas onto coil
            data = (self.data["fiducial_delta"] + self.data["fiducial_target"]).sel(
                target=target
            )
            # convert to pandas DataFrame
            dataframe = pandas.concat(
                {coil: data.sel(coil=coil).to_pandas() for coil in data.coil.data}
            )
            dataframe.index.set_names(["coil", "target"], inplace=True)
            # save projection transform to instance
            self.fiducial_ilis = ilis
            # project to ilis plane
            dataframe = ilis.project(dataframe)
            for i, (_, frame) in enumerate(dataframe.groupby("coil")):
                data.data[i] = frame.values  # reassign data
            # update fiducial deltas with tagets aligned to ilis midplane
            self.data["fiducial_delta"].loc[:, target] = data - self.data[
                "fiducial_target"
            ].sel(target=target)

        if hasattr(self.dataset, "variance"):
            self.data["fiducial_variance"] = (
                ("coil", "target", "space"),
                np.stack(
                    [
                        self.dataset.variance[index].to_numpy(float)
                        for index in self.dataset.variance
                    ],
                    axis=0,
                ),
            )
        if hasattr(self.dataset, "sectors"):
            self.data.coords["sector"] = list(self.dataset.sectors)
            self.data["coils"] = (
                ("sector", "coil_index"),
                np.array(list(self.dataset.sectors.values())),
            )
            self.data["filename"] = (
                "sector",
                [SectorData(sector).filename for sector in self.dataset.sectors],
            )
            self.data["version"] = (
                "sector",
                [SectorData(sector).version for sector in self.dataset.sectors],
            )

        self.data["centerline_delta"] = xarray.DataArray(
            0.0,
            coords=[
                ("coil", self.data.coil.values),
                ("arc_length", self.data.arc_length.values),
                ("space", self.data.space.values),
            ],
        )
        for coil_index in range(self.data.sizes["coil"]):
            for space_index in range(self.data.sizes["space"]):
                self.data["centerline_delta"][coil_index, :, space_index] = (
                    self.load_gpr(coil_index, space_index)
                )
        self.data.attrs["source"] = self.dataset.source

    def load_gpr(self, coil_index, space_index):
        """Return Gaussian Process regression."""
        match self.variance:
            case "file":
                variance = self.data.fiducial_variance[coil_index, :, space_index].data
            case float():
                variance = self.variance
            case _:
                raise ValueError(f"variance {self.variance} not file or float")
        self.gpr = GaussianProcessRegressor(
            self.data.target_length[coil_index], variance=variance
        )
        return self.gpr.evaluate(
            self.data.arc_length, self.data.fiducial_delta[coil_index, :, space_index]
        )

    def plot_gpr(self, coil_index, space_index):
        """Plot Gaussian Process regression."""
        self.load_gpr(coil_index, space_index)
        self.gpr.plot()

    def plot_gpr_array(self, coil_index, stage, axes=None, **kwargs):
        """Plot gpr array."""
        self.axes = self.set_axes(
            "1d", nrows=3, ncols=1, sharex=True, sharey=True, aspect=0.7, axes=axes
        )
        for space_index, coord in enumerate("xyz"):
            self.load_gpr(coil_index, space_index)
            self.gpr.plot(stage, axes=self.axes[space_index], text=False, **kwargs)
            self.axes[space_index].set_ylabel(rf"$\Delta{{{coord}}}$ mm")
        self.axes[-1].set_xlabel("arc length")
        self.axes[0].legend(loc="center", bbox_to_anchor=(0, 1.05, 1, 0.1), ncol=2)

    @staticmethod
    def fiducials():
        """Return fiducial coordinates."""
        return pandas.DataFrame(
            index=list(string.ascii_uppercase[:8]),
            columns=["x", "y", "z"],
            data=[
                [2713.7, 0.0, -3700.0],
                [2713.7, 0.0, 3700.0],
                [5334.4, 0.0, 6296.4],
                [8980.4, 0.0, 4437.0],
                [9587.6, 0.0, -3695.0],
                [3399.7, 0.0, -5598.0],
                [10733.0, 0.0, 0.0],
                [2713.7, 0.0, 0.0],
            ],
        )

    def plot(self, factor=250, axes=None):
        """Plot fiudicial points on coil cenerline."""
        self.axes = self.set_axes("2d", nrows=1, ncols=2, sharey=True, axes=axes)
        for j in range(2):
            self.axes[j].plot(
                self.data.centerline_target[0, :, 0],
                self.data.centerline_target[0, :, 2],
                "gray",
                ls="--",
            )
        limits = self.axes_limit
        color = [0, 0]
        for coil_index in range(self.data.sizes["coil"]):
            j = 0 if self.data.origin[coil_index] == "EU" else 1
            self.axes[j].plot(
                self.data.centerline_target[coil_index, :, 0]
                + factor * self.data.centerline_delta[coil_index, :, 0],
                self.data.centerline_target[coil_index, :, 2]
                + factor * self.data.centerline_delta[coil_index, :, 2],
                color=f"C{color[j]}",
                label=f"{self.data.coil[coil_index].values:02d}",
            )
            self.axes[j].plot(
                self.data.fiducial_target[coil_index, :, 0]
                + factor * self.data.fiducial_delta[coil_index, :, 0],
                self.data.fiducial_target[coil_index, :, 2]
                + factor * self.data.fiducial_delta[coil_index, :, 2],
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

    def coil_index(self, coil: int):
        """Return coil index."""
        return list(self.data.coil).index(coil)

    def plot_single(self, coil_index, stage=3, factor=500, axes=None, color="gray"):
        """Plot single fiducial curve."""
        self.set_axes(
            "2d",
            axes,
            aspect=1,
            nrows=1,
            ncols=2,
            sharey=True,
            width_ratios=[3, 1],
        )
        for ax, (i, j) in enumerate(zip((0, 1), (2, 2))):
            self.axes[ax].plot(
                self.data.centerline_target[coil_index, :, i],
                self.data.centerline_target[coil_index, :, j],
                "gray",
                ls="--",
            )
        limits = self.axes_limit
        for ax, (i, j) in enumerate(zip((0, 1), (2, 2))):
            for fiducial in self.data.fiducial_target:
                self.axes[ax].plot(fiducial[:, i], fiducial[:, j], "ko")
                if ax == 0:
                    for x, y, label in zip(
                        fiducial[:, i], fiducial[:, j], fiducial.target.values
                    ):
                        self.axes[ax].text(x, y, f" {label}")
            if stage > 0:
                self.axes[ax].plot(
                    self.data.fiducial_target[coil_index, :, i]
                    + factor * self.data.fiducial_delta[coil_index, :, i],
                    self.data.fiducial_target[coil_index, :, j]
                    + factor * self.data.fiducial_delta[coil_index, :, j],
                    "o",
                    color=color,
                    alpha=0.5,
                )
            if stage > 1:
                self.axes[ax].plot(
                    self.data.centerline_target[coil_index, :, i]
                    + factor * self.data.centerline_delta[coil_index, :, i],
                    self.data.centerline_target[coil_index, :, j]
                    + factor * self.data.centerline_delta[coil_index, :, j],
                    color=color,
                )
            if stage > 2:
                gpr_fiducial = (
                    self.data.centerline_target[coil_index]
                    + factor * self.data.centerline_delta[coil_index]
                )

                self.axes[ax].plot(
                    gpr_fiducial[self.data.target_index[coil_index], i],
                    gpr_fiducial[self.data.target_index[coil_index], j],
                    "d",
                    markersize=12,
                    color=color,
                    alpha=0.5,
                )
        limits[0]["x"] = [1500, 11000]
        limits[1]["x"] = [-1100, 1100]
        limits[0]["y"] = [-8000, 8000]
        limits[1]["y"] = [-8000, 8000]
        self.axes_limit = limits
        # self.axes[0].set_title(f"TF#{self.data.coil[coil_index].data} {self.phase}")


if __name__ == "__main__":
    phase = "FAT supplier"
    phase = "SSAT BR"

    fiducial = FiducialData(
        sectors={7: [8, 9]}, phase=phase, variance="file", fill=False
    )

    # fiducial = FiducialData(fiducial="Sector", phase=phase, fill=False, variance=0.09)

    # fiducial.load_fiducials()
    # fiducial.plot()

    coil = 8
    coil_index = fiducial.coil_index(coil)

    fiducial.plot_single(coil_index, 3, 500)

    # fiducial.fig.tight_layout(pad=0.5)
    # fiducial.savefig("single")

    # fiducial.plot_gpr(1, 0)

    # fiducial.plot_gpr_array(coil_index, 3)
    # fiducial.fig.tight_layout(pad=0.5)
    # fiducial.savefig("gpr_array")

    # fiducial.plot()
    # fiducial.fig.tight_layout(pad=0)
    # fiducial.savefig("fiducial")

    """
    plotter = pv.Plotter()
    fiducial.mesh['delta'] *= 1e3
    fiducial.warp(0.5, plotter=plotter)
    fiducial.label_coils(plotter)
    plotter.show_axes()
    plotter.show()
    """

    #
