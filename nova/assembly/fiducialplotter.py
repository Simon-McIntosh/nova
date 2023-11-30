"""Manage plotting methods for fiducial data."""
from dataclasses import dataclass, field, InitVar
from functools import cached_property
from typing import ClassVar

import matplotlib.pyplot as plt
import seaborn as sns
import xarray

from nova.assembly.transform import Rotate


@dataclass
class FiducialPlotter:
    """Plot fidicual fit in cylindrical coordinates."""

    cartisean_data: InitVar[xarray.Dataset]
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)
    factor: float = 750
    fiducial_labels: bool = True
    rotate: Rotate = field(init=False, default_factory=Rotate)

    color: ClassVar[dict[str, str]] = dict(
        fit="C1", fit_target="C0", reference="C4", reference_target="C6"
    )
    marker: ClassVar[dict[str, str]] = dict(
        fit="X", fit_target="d", reference="X", reference_target="d"
    )

    def __post_init__(self, data: xarray.Dataset):
        """Transform data to cylindrical coordinates."""
        self.extract(data)

    def __call__(self, label: str = "target", stage: int = 2, coil_index=range(2)):
        """Plot fiducial and centerline fits."""
        if label == "target":
            return self.target()
        if stage > 0:
            self.fiducial(label, coil_index=coil_index)
        if stage > 1:
            self.fiducial(f"{label}_target", coil_index=coil_index)
            self.centerline(label, coil_index=coil_index)

    def extract(self, cartisean_data: xarray.Dataset):
        """Extract cartisean data and map to cylindrical coordinates."""
        self.data = xarray.Dataset()
        self.data["fiducial_target"] = Rotate.to_cylindrical(
            cartisean_data.fiducial_target
        )
        self.data["centerline_target"] = Rotate.to_cylindrical(
            cartisean_data.centerline_target
        )
        for attr in ["reference", "fit"]:
            if attr not in cartisean_data:
                continue
            self.data[attr] = (
                self.rotate.to_cylindrical(cartisean_data[attr]) - self.data.fiducial
            )
            for norm in ["fiducial", "centerline"]:
                self.data[f"{attr}_{norm}"] = (
                    self.rotate.to_cylindrical(cartisean_data[f"{attr}_{norm}"])
                    - self.data[norm]
                )
            self.data[f"{attr}_centerline_sample"] = (
                cartisean_data[f"{attr}_centerline_sample"] - self.data["centerline"]
            )
        self.data.fiducial[..., 1] = 0
        self.data.centerline[..., 1] = 0

    @cached_property
    def axes(self):
        """Return axes instance."""
        axes = plt.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[2.5, 1]))[
            1
        ]
        axes[0].set_xlabel("radius")
        axes[0].set_ylabel("height")
        axes[1].set_xlabel("toroidal")
        for i in range(2):
            axes[i].axis("equal")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            sns.despine()
        return axes

    def plot_box(self, data_array: xarray.DataArray):
        """Plot bounding box around fiducial targets."""

    '''
    def fiducial(self, coil_index=range(2)):
        """Plot fiducial fiducial targets."""
        for i in coil_index:
            self.axes[0].plot(
                self.data.centerline[i, :, 0],
                self.data.centerline[i, :, 2],
                "--",
                color="gray",
            )
            self.axes[1].plot(
                self.data.centerline[i, :, 1],
                self.data.centerline[i, :, 2],
                "--",
                color="gray",
            )
            self.axes[0].plot(
                self.data.fiducial[i, :, 0],
                self.data.fiducial[i, :, 2],
                "o",
                color="gray",
            )
            self.axes[1].plot(
                self.data.fiducial[i, :, 1],
                self.data.fiducial[i, :, 2],
                "o",
                color="gray",
            )
        if self.fiducial_labels:
            for radius, height, label in zip(
                self.data.fiducial[0, :, 0],
                self.data.fiducial[0, :, 2],
                self.data.fiducial.values,
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
    '''

    def delta(self, label: str):
        """Return displacment delta multiplied by scale factor."""
        return self.factor * self.data[f"{label}"]

    def fiducial(self, label: str, coil_index=range(2)):
        """Plot fiducial deltas."""
        color = self.color.get(label, self.color[label.split("_")[0]])
        marker = self.marker[label]
        for i in coil_index:
            delta = self.delta(label)
            self.axes[0].plot(
                self.data.fiducial[i, :, 0] + delta[i, :, 0],
                self.data.fiducial[i, :, 2] + delta[i, :, 2],
                color + marker,
            )
            self.axes[1].plot(
                self.data.fiducial[i, :, 1] + delta[i, :, 1],
                self.data.fiducial[i, :, 2] + delta[i, :, 2],
                color + marker,
            )

    def centerline(self, label: str, coil_index=range(2), samples=True):
        """Plot gpr centerline."""
        color = self.color[f"{label}_fiducial"]
        attr = f"{label}_centerline"
        if samples:
            attr += "_sample"
        for i in coil_index:
            delta = self.delta(attr)
            self.axes[0].plot(
                self.data.centerline[i, :, 0] + delta[i, :, 0],
                self.data.centerline[i, :, 2] + delta[i, :, 2],
                color=color,
            )
            self.axes[1].plot(
                self.data.centerline[i, :, 1] + delta[i, :, 1],
                self.data.centerline[i, :, 2] + delta[i, :, 2],
                color=color,
            )
