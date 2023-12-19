"""Manage signal methods."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import minmax_scale
import scipy.signal
import xarray

from nova.graphics.plot import Plot
from nova.geometry.rdp import rdp
from nova.imas.database import Database, Ids, IdsEntry
from nova.imas.equilibrium import EquilibriumData
from nova.imas.metadata import Metadata


@dataclass
class Select:
    """Select subset of data based on coordinate."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

    def attrs(self, coord: str):
        """Return attribute list selected according to coord."""
        if coord[0] == "~":
            return [
                attr
                for attr, value in self.data.items()
                if coord[1:] not in value.coords or len(value.shape) > 1
            ]
        return [
            attr
            for attr, value in self.data.items()
            if coord in value.coords and len(value.shape) == 1
        ]

    def select(self, coord: str, data=None, dtype=None):
        """Return data subset including all data variables with coord."""
        if data is None:
            data = self.data
        attrs = self.attrs(coord)
        if dtype is None:
            return data[attrs]
        attrs = [attr for attr in attrs if self.data[attr].data.dtype == dtype]
        return data[attrs]


@dataclass
class Defeature:
    """Defeature dataset using a clustered RDP algoritum."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    epsilon: float = 1e-3
    cluster: int | float | None = None
    features: list[str] | None = None

    def __post_init__(self):
        """Extract feature list if None."""
        self.check_features()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def check_features(self):
        """Check features, update it None."""
        match self.features:
            case None:
                self.features = [
                    attr
                    for attr, value in self.data.items()
                    if value.coords.dims == ("time",)
                ]
            case list():
                assert np.all(
                    self.data[attr].coords.dims == ("time",) for attr in self.features
                )
            case _:
                raise TypeError(f"features {self.feaures} not list")

    @cached_property
    def time(self):
        """Return time vector with shape (n, 1)."""
        return np.copy(self.data.time.data[:, np.newaxis])

    def defeature(self):
        """Return clustered turning point dataset."""
        indices = []
        index = np.arange(self.data.sizes["time"])
        for attr in self.features:
            array = np.c_[self.time, minmax_scale(self.data[attr].data)]
            mask = rdp(array, self.epsilon, return_mask=True)
            indices.extend(index[mask])
        indices = np.unique(indices)
        if self.cluster is not None:
            indices = self._cluster(indices)
        return self.data.isel({"time": indices})

    def _cluster(self, indices):
        """Apply DBSCAN clustering algorithum to indices."""
        time = self.time[indices]
        clustering = DBSCAN(eps=self.cluster, min_samples=1).fit(time)
        labels = np.unique(clustering.labels_)
        centroid = np.zeros(len(labels), int)
        label_index = np.arange(len(indices))
        for i, label in enumerate(labels):
            centroid[i] = int(np.mean(label_index[label == clustering.labels_]))
        return indices[centroid]


@dataclass
class Sample(Plot, Defeature, Select):
    """Re-sample signal."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    dtime: int | float | None = None
    savgol: tuple[int, int] | None = (3, 1)
    epsilon: float = 0.25
    cluster: int | float | None = None
    features: list[str] = field(
        default_factory=lambda: [
            "minor_radius",
            "elongation",
            "triangularity_upper",
            "triangularity_lower",
            "triangularity_inner",
            "triangularity_outer",
            "squareness_upper_inner",
            "squareness_upper_outer",
            "squareness_lower_inner",
            "squareness_lower_outer",
            "li_3",
            "beta_normal",
            "ip",
        ]
    )
    samples: dict[str, xarray.Dataset] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Interpolate data onto uniform time-base and resample."""
        self["source"] = self.data
        self.clip("li_3", 0)
        self.clip("ip", 1e-5)
        if self.dtime is not None:
            self.interpolate()
            self.resample()
        self.smooth()
        self.defeature()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def __getitem__(self, attr: str) -> xarray.Dataset:
        """Return dataset from samples dict."""
        return self.samples[attr]

    def __setitem__(self, attr: str, data: xarray.Dataset):
        """Set item in profiles dict."""
        self.data = data
        self.samples[attr] = data

    def clip(self, attr: str, value: float | str):
        """Select data as abs(attr) > value."""
        time = self.data.time[abs(self.data[attr]) > value]
        self["clip"] = self.data.sel({"time": time})

    @cached_property
    def minimum_timestep(self) -> float:
        """Return minimum timestep present in source data."""
        return np.diff(self.data.time).min()

    @property
    def factor(self):
        """Return re-sample factor."""
        match self.dtime:
            case int() if self.dtime < 0:
                return -self.dtime / float(self.data.sizes["time"])
            case int() | float() if self.dtime > 0:
                return self.minimum_timestep / self.dtime
            case _:
                raise ValueError(
                    f"dtime {self.dtime} is " "not a negative int or float"
                )

    @property
    def updown(self) -> tuple[int, int]:
        """Return up and downsampling factors."""
        match self.factor:
            case float(factor) if factor == 0:
                return 1, 1
            case float(factor) if factor > 1:
                return int(10 * round(factor, 1)), 10
            case float(factor) if factor < 1:
                return 10, int(10 * round(1 / factor, 1))
        raise ValueError(f"invalid sample factor {self.factor}")

    def interpolate(self):
        """Interpolate data onto uniform time-base."""
        time = np.arange(self.data.time[0], self.data.time[-1], self.minimum_timestep)

        self["uniform"] = self.data.interp({"time": time}).assign_coords(
            {"itime": range(len(time))}
        )

    def resample(self):
        """Return dataset re-sampled using a polyphase filter."""
        updown = self.updown
        factor = updown[0] / updown[1]
        ntime = int(np.ceil(self["uniform"].sizes["time"] * factor))
        time = np.linspace(self.data.time[0], self.data.time[-1], ntime)
        time_sample = xarray.Dataset(coords={"time": time})
        time_sample.coords["itime"] = "time", np.arange(len(time))
        for attr, value in self.select("time", self["uniform"]).items():
            dims = value.coords.dims
            value = scipy.signal.resample_poly(value, *updown, padtype="line")
            time_sample[attr] = dims, value
        self["sample"] = xarray.merge(
            [self.select("~time", self["uniform"]), time_sample]
        )

    def smooth(self):
        """Smooth signal using savgol filter."""
        if self.savgol is None:
            return
        savgol = xarray.Dataset(coords={"time": self.data.time})
        for attr, value in self.select("time", self.data, float).items():
            dims = value.coords.dims
            value = scipy.signal.savgol_filter(value, *self.savgol, axis=0)
            savgol[attr] = dims, value
        self["smooth"] = xarray.merge(
            [
                self.select("~time", self.data),
                self.select("time", self.data, int),
                savgol,
            ]
        )

    def defeature(self):
        """Defeature sample waveform using rdp algorithum."""
        self["rdp"] = super().defeature()

    def _plot_attr(self, sample: str, attr: str, scale=True, **kwargs):
        """Plot single attribute waveform."""
        self.axes = kwargs.get("axes", None)
        data = self[sample]
        value = data[attr]
        if scale:
            value = minmax_scale(value, axis=0)
        self.axes.plot(data.time, value, **kwargs)

    def plot(self, attrs=None, scale=False, rdp=True):
        """Plot source, interpolated, and sampled datasets."""
        if attrs is None:
            attrs = [attr for attr in self.features if attr != "ip"]
        if isinstance(attrs, str):
            attrs = [attrs]
        self.set_axes("1d")
        for i, attr in enumerate(attrs):
            dims = self.data[attr].coords.dims
            if "time" not in dims or len(dims) != 1:
                continue
            self._plot_attr(
                "clip", attr, ls="-", color=f"C{i}", lw=2.0, label=attr, scale=scale
            )
            if self.dtime is not None:
                self._plot_attr("sample", attr, ls="-", color="gray", lw=1, scale=scale)
            if self.savgol is not None:
                self._plot_attr("smooth", attr, ls="-", color="k", lw=0.5, scale=scale)
            if rdp:
                self._plot_attr(
                    "rdp",
                    attr,
                    ls="-",
                    marker="o",
                    color="k",
                    lw=1.5,
                    ms=6,
                    zorder=-10,
                    scale=scale,
                    mfc="k",
                    mec=f"C{i}",
                )
        self.axes.legend(ncol=2)
        self.axes.set_xlabel("time s")
        ylabel = "value"
        if scale:
            ylabel = f"normalized {ylabel}"
        self.axes.set_ylabel(ylabel)

    def pulse_schedule_ids(self) -> Ids:
        """Write sample data to a pulse schedule IDS."""
        ids_entry = IdsEntry(name="pulse_schedule")
        self.update_metadata(ids_entry, provenance=[self.data.attrs["pulse_schedule"]])
        ids_entry.ids_data.time = self.data.time.data
        with ids_entry.node("flux_control.*.reference.data"):
            ids_entry["i_plasma"] = self.data.ip.data
            ids_entry["loop_voltage"] = self.data.psi_boundary.data
            for attr in ["li_3", "beta_normal"]:
                ids_entry[attr] = self.data[attr].data

        with ids_entry.node("position_control.geometric_axis." "*.reference.data"):
            for i, attr in enumerate("rz"):
                ids_entry[attr] = self.data.geometric_axis[:, i].data

        with ids_entry.node("position_control.*.reference.data"):
            for attr in [
                "minor_radius",
                "elongation",
                "triangularity_upper",
                "triangularity_lower",
            ]:
                ids_entry[attr] = self.data[attr].data
            # TODO fix IDS
            for tmp_attr, attr in zip(
                ["elongation_upper", "elongation_lower"],
                ["triangularity_outer", "triangularity_inner"],
            ):
                ids_entry[tmp_attr] = self.data[attr].data

        ids_entry.resize("position_control.x_point", 1)
        with ids_entry.node("position_control.x_point:*.reference.data"):
            ids_entry["r", 0] = self.data.x_point[:, 0].data
            ids_entry["z", 0] = self.data.x_point[:, 1].data

        ids_entry.resize("position_control.strike_point", 2)
        with ids_entry.node("position_control.strike_point:*.reference.data"):
            for i in range(2):
                ids_entry["r", i] = self.data.strike_point[:, i, 0].data
                ids_entry["z", i] = self.data.strike_point[:, i, 1].data
        return ids_entry.ids_data

    def equilibrium_ids(self) -> Ids:
        """Write sample data to a equilibrium IDS."""
        ids_entry = IdsEntry(name="equilibrium")
        self.update_metadata(ids_entry, provenance=[self.data.attrs["equilibrium"]])
        ids_entry.ids_data.time = self.data.time.data
        ids_entry.ids_data.time_slice.resize(self.data.sizes["time"])
        with ids_entry.node("time_slice:global_quantities.*"):
            for attr in ["ip", "li_3", "beta_normal"]:
                ids_entry[attr, :] = self.data[attr].data

        with ids_entry.node("time_slice:boundary_separatrix.*"):
            ids_entry["type", :] = self.data["boundary_type"].data
            ids_entry["psi", :] = self.data["psi_boundary"].data
            for attr in [
                "minor_radius",
                "elongation",
                "triangularity_upper",
                "triangularity_lower",
                "squareness_upper_inner",
                "squareness_upper_outer",
                "squareness_lower_inner",
                "squareness_lower_outer",
            ]:
                ids_entry[attr, :] = self.data[attr].data
            # TODO fix IDS
            for tmp_attr, attr in zip(
                ["elongation_upper", "elongation_lower"],
                ["triangularity_outer", "triangularity_inner"],
            ):
                ids_entry[tmp_attr, :] = self.data[attr].data

        with ids_entry.node("time_slice:boundary_separatrix." "geometric_axis.*"):
            for i, attr in enumerate("rz"):
                ids_entry[attr, :] = self.data.geometric_axis[:, i].data

        with ids_entry.node("time_slice:boundary_separatrix.x_point:*"):
            for itime in range(self.data.sizes["time"]):
                if self.data.x_point_number.data[itime] == 1:
                    ids_entry["r", itime] = self.data.x_point.data[itime, 0]
                    ids_entry["z", itime] = self.data.x_point.data[itime, 1]

        with ids_entry.node("time_slice:boundary_separatrix.strike_point:*"):
            for itime in range(self.data.sizes["time"]):
                for i in range(self.data.sizes["strike_point_index"])[::-1]:
                    for j, attr in enumerate("rz"):
                        ids_entry[attr, itime, i] = self.data.strike_point.data[
                            itime, i, j
                        ]

        # include profile data - to remove in future
        with ids_entry.node("time_slice:profiles_1d.*"):
            for itime in range(self.data.sizes["time"]):
                ids_entry["psi", itime] = self.data.psi1d.data[itime]
                for attr in ["dpressure_dpsi", "f_df_dpsi"]:
                    ids_entry[attr, itime] = self.data[attr].data[itime]
        return ids_entry.ids_data

    def update_metadata(self, ids_entry: IdsEntry, provenance=None):
        """Generate ids_entry and add metadata."""
        metadata = Metadata(ids_entry.ids_data)
        comment = "Feature preserving reduced order waveforms"
        metadata.put_properties(comment, homogeneous_time=1, provenance=provenance)
        code_parameters = {
            attr: getattr(self, attr)
            for attr in ["dtime", "savgol", "epsilon", "cluster", "features"]
        }
        metadata.put_code(code_parameters)

    def write_ids(self, **ids_attrs):
        """Write sample data to pulse_schedule ids."""
        match ids_attrs["name"]:
            case "equilibrium":
                ids_data = self.equilibrium_ids()
            case "pulse_schedule":
                ids_data = self.pulse_schedule_ids()
            case _:
                raise NotImplementedError(
                    "write_ids not implemented for " f'ids_name {ids_attrs["name"]}'
                )
        if ids_attrs["occurrence"] is None:
            ids_attrs["occurrence"] = Database(**ids_attrs).next_occurrence()
        ids_entry = IdsEntry(ids_data=ids_data, **ids_attrs)
        ids_entry.put_ids()


if __name__ == "__main__":
    pulse, run = 135013, 2
    pulse, run = 105028, 1

    pulse, run = 135013, 2
    # pulse, run = 105050, 2
    # pulse, run = 135001, 7

    equilibrium = EquilibriumData(pulse, run, occurrence=0)
    sample = Sample(
        equilibrium.data, features=["ip", "x_point_number", "psi_axis"], cluster=1.5
    )

    sample.plot()
    sample.plot(attrs=["ip"])

    # print(sample.data)

    sample.write_ids(**equilibrium.ids_attrs | {"occurrence": 1})
    sample.plot(
        [
            "minor_radius",
            "elongation",
            "triangularity_upper",
            "triangularity_lower",
            "triangularity_inner",
            "triangularity_outer",
        ]
    )

    """
    sample.plot(
        [
            "squareness_upper_inner",
            "squareness_upper_outer",
            "squareness_lower_inner",
            "squareness_lower_outer",
        ]
    )
    """

    # sample.plot('ip')
