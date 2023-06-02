"""Access imas 1d profile data."""
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
import xarray

from nova.database.netcdf import netCDF
from nova.imas.connect import ScenarioDatabase
from nova.imas.equilibrium import EquilibriumData
from nova.graphics.plot import Plot1D


@dataclass
class EnsembleAttrs:
    """Specify non-default attributes for Profile class."""

    workflow: str
    name: str | None = "equilibrium"
    attrs: list[str] = field(default_factory=lambda: ["f_df_dpsi", "dpressure_dpsi"])


@dataclass
class Ensemble(netCDF, Plot1D, EnsembleAttrs):
    """Manage workflow ensemble equilibrium 1d profile data."""

    filename: str = "ensemble"
    datapath: str = "data/Imas"
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Set filepath."""
        self.group = f'{self.name}/{self.workflow.replace("-", "_")}'
        self.set_path(self.datapath)
        try:
            self.load()
        except (FileNotFoundError, OSError):
            self.build()

    def build(self):
        """Build dataset."""
        frame = ScenarioDatabase().load_frame("workflow", self.workflow)
        frame = self._format_columns(frame)
        data = xarray.Dataset.from_dataframe(frame)
        self.data = data.set_coords(frame.columns)
        self.data = xarray.merge(
            [self.data, self._load_attrs()], combine_attrs="drop_conflicts"
        )
        self.store()

    @staticmethod
    def _format_columns(frame):
        """Return frame with formated columns."""
        columns = [col if "[" in col else col.lower() for col in frame.columns]
        columns = [col.split("[")[0] for col in columns]
        columns = {old: new for old, new in zip(frame, columns)}
        return frame.rename(columns=columns)

    def _load_attrs(self):
        """Retrun concatinated dataset from imas equilibrium workflow."""
        data = []
        for i, (pulse, run) in enumerate(zip(self.data.pulse.data, self.data.run.data)):
            eq_data = EquilibriumData(pulse, run).data[self.attrs]
            _isnull = eq_data.isnull().any()
            if any([getattr(_isnull, attr) for attr in _isnull]):
                warn(f"\nskipping {pulse}:{run} due to nans in dataset")
                continue
            eq_data.coords["pulse_index"] = (
                "time",
                i * np.ones_like(eq_data.time, int),
            )
            data.append(eq_data)
        return xarray.concat(data, "time", combine_attrs="drop_conflicts")

    def plot(self, attr: str, **kwargs):
        """Plot ensemble attribute."""
        self.axes = kwargs.pop("axes", None)
        _color = kwargs.pop("color", None)
        self.data.load()
        for i in self.data.index.data:
            index = self.data.subindex == i
            data = self.data[attr][index].data[::100]
            color = f"C{i%10}" if _color is None else _color
            self.axes.plot(
                self.data.psi_norm.data,
                data.T,
                color=color,
                label=self.data.reference.data[i],
                **kwargs,
            )
            self.axes.legend()


if __name__ == "__main__":
    ens = Ensemble("DINA-IMAS")
    ens.build()

    # ens.plot('f_df_dpsi')
