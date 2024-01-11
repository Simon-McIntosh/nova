"""Manage plotting methods for fiducial data."""
from dataclasses import dataclass, field, InitVar
from typing import ClassVar

import numpy as np
import xarray

from nova.assembly.transform import Rotate
from nova.graphics.plot import Plot


@dataclass
class FiducialPlotter(Plot):
    """Plot fidicual fit in cylindrical coordinates."""

    cartisean_data: InitVar[xarray.Dataset]
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)
    factor: float = 750
    fiducial_labels: bool = True

    color: ClassVar[dict[str, str]] = {
        "": "C0",
        "gpr": "C1",
        "fit": "C4",
        "gpr_fit": "C6",
    }
    marker: ClassVar[dict[str, str]] = {"": "o", "fit": "X", "gpr": "d", "gpr_fit": "X"}

    def __post_init__(self, data: xarray.Dataset):
        """Transform data to cylindrical coordinates."""
        self.rotate(data)
        self.reset_axes()

    def reset_axes(self):
        """Extend Plot.set_axes."""
        self.axes = super().set_axes(
            "plan",
            nrows=1,
            ncols=2,
            sharey=True,
            gridspec_kw=dict(width_ratios=[2.5, 1]),
        )
        self.axes[0].set_xlabel("radius")
        self.axes[0].set_ylabel("height")
        self.axes[1].set_xlabel("toroidal")

    def __call__(self, post_fix: str, stage: int = 2, coil_index=0):
        """Plot fiducial and centerline fits."""

        if stage > 0:
            self.fiducial(post_fix, coil_index=coil_index)
            self.centerline(post_fix, coil_index=coil_index)
        # if stage > 1:
        #    self.fiducial(f"{label}_target", coil_index=coil_index)
        #    self.centerline(label, coil_index=coil_index)

    @staticmethod
    def join(name: str, post_fix: str):
        """Return variable name with post_fix if set."""
        if post_fix:
            return "_".join([name, post_fix])
        return name

    def rotate(self, cartisean_data: xarray.Dataset):
        """Rotate cartisean data to cylindrical coordinates."""
        self.data = xarray.Dataset()
        for name in ["fiducial", "centerline"]:
            target = "_".join([name, "target"])
            self.data[target] = Rotate.to_cylindrical(cartisean_data[target])
            for post_fix in ["", "gpr", "fit", "gpr_fit"]:
                attr = self.join(name, post_fix)
                self.data[attr] = (
                    Rotate.to_cylindrical(cartisean_data[attr]) - self.data[target]
                )

    def plot_box(self, data_array: xarray.DataArray):
        """Plot bounding box around fiducial targets."""

    def target(self, coil_index):
        """Plot fiducial fiducial targets."""
        self.axes[0].plot(
            self.data.centerline_target[:, 0],
            self.data.centerline_target[:, 2],
            "--",
            color="gray",
        )
        self.axes[1].plot(
            self.data.centerline_target[:, 1],
            self.data.centerline_target[:, 2],
            "--",
            color="gray",
        )
        self.axes[0].plot(
            self.data.fiducial_target[coil_index, :, 0],
            self.data.fiducial_target[coil_index, :, 2],
            "o",
            color="gray",
        )
        self.axes[1].plot(
            self.data.fiducial_target[coil_index, :, 1],
            self.data.fiducial_target[coil_index, :, 2],
            "o",
            color="gray",
        )
        if self.fiducial_labels:
            for radius, height, label in zip(
                self.data.fiducial_target[coil_index, :, 0],
                self.data.fiducial_target[coil_index, :, 2],
                self.data.target.values,
            ):
                self.axes[0].text(
                    radius,
                    height,
                    f"{label} ",
                    ha="right",
                    va="center",
                    color="gray",
                    fontsize="x-large",
                    zorder=-10,
                )
        self.axes[0].plot([800, 12000], [-8000, 8000], "w.")
        self.axes[1].plot([-2500, 2500], [-8000, 8000], "w.")

    def delta(self, attr: str):
        """Return displacment delta multiplied by scale factor."""
        return self.factor * self.data[attr]

    def fiducial(self, post_fix: str, coil_index=0):
        """Plot fiducial deltas."""
        color = self.color[post_fix]
        marker = self.marker[post_fix]
        delta = self.delta(self.join("fiducial", post_fix))
        for i in np.atleast_1d(coil_index):
            self.axes[0].plot(
                self.data.fiducial_target[i, :, 0] + delta[i, :, 0],
                self.data.fiducial_target[i, :, 2] + delta[i, :, 2],
                color + marker,
            )
            self.axes[1].plot(
                self.data.fiducial_target[i, :, 1] + delta[i, :, 1],
                self.data.fiducial_target[i, :, 2] + delta[i, :, 2],
                color + marker,
            )

    def centerline(self, post_fix: str, coil_index=0, samples=True):
        """Plot gpr centerline."""
        color = self.color[post_fix]
        attr = self.join("centerline", post_fix)
        # if samples:
        #    attr += "_sample"
        target = self.data["centerline_target"]
        delta = self.delta(attr)
        for i in np.atleast_1d(coil_index):
            self.axes[0].plot(
                target[:, 0] + delta[i, :, 0],
                target[:, 2] + delta[i, :, 2],
                color=color,
            )
            self.axes[1].plot(
                target[:, 1] + delta[i, :, 1],
                target[:, 2] + delta[i, :, 2],
                color=color,
            )
