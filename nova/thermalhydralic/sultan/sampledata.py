"""Manage sultan sample data."""

from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

import pandas
import numpy as np
import scipy.signal
import CoolProp.CoolProp as CoolProp

from nova.thermalhydralic.sultan.heatindex import HeatIndex
from nova.thermalhydralic.sultan.sourcedata import SourceData
from nova.thermalhydralic.sultan.trial import Trial


@dataclass
class SampleData:
    """Manage sample dataframe."""

    sourcedata: SourceData = field(repr=False)
    _lowpass_filter: Union[bool, list[bool]] = True
    _raw: pandas.DataFrame = field(init=False, repr=False)
    _lowpass: pandas.DataFrame = field(init=False, repr=False)
    _heatindex: HeatIndex = field(init=False, repr=False)
    reload: SimpleNamespace = field(
        init=False, repr=False, default_factory=SimpleNamespace
    )

    def __post_init__(self):
        """Init data pipeline."""
        self.reload.__init__(
            raw=True, lowpass=True, heatindex=True, offset=True, waveform=True
        )

    @property
    def lowpass_filter(self):
        """Return low-pass filter flag."""
        if isinstance(self._lowpass_filter, bool):
            return self._lowpass_filter
        return self._lowpass_filter[0]

    @lowpass_filter.setter
    def lowpass_filter(self, lowpass_filter):
        self._lowpass_filter = lowpass_filter

    def __call__(self, lowpass_filter=True):
        """Store current filter flag and activate temporary value."""
        self._lowpass_filter = [lowpass_filter, self.lowpass_filter]
        return self

    def __enter__(self):
        """Enter dunner."""

    def __exit__(self, exception_type, exception_value, traceback):
        """Reset filter flag."""
        self.lowpass_filter = self._lowpass_filter[1]

    def _reload(self):
        """Propagate reload flags."""
        self.sourcedata._reload()
        if self.sourcedata.reload.sampledata:
            self.reload.raw = True
            self.reload.lowpass = True
            self.reload.heatindex = True
            self.reload.offset = True
            self.reload.waveform = True
            self.sourcedata.reload.sampledata = False

    @property
    def sultandata(self):
        """Return source data."""
        return self.sourcedata.data

    @property
    def side(self):
        """Return sample side, read-only."""
        return self.sourcedata.side

    @property
    def frequency(self):
        """Return sample frequency, read-only."""
        return self.sourcedata.frequency

    @property
    def excitation_field_rate(self):
        """Return excitation field rate of change."""
        return self.sourcedata.excitation_field_rate

    @staticmethod
    def _initialize_dataframe():
        """
        Return calclation dataframe.

        Returns
        -------
        dataframe : pandas.DataFrame
            Empty dataframe with time index and default columns names.

        """
        variables = [
            ("t", "s"),
            ("mdot", "kg/s"),
            ("Ipulse", "A"),
            ("Tin", "K"),
            ("Tout", "K"),
            ("Pin", "Pa"),
            ("Pout", "Pa"),
            ("hin", "J/Kg"),
            ("hout", "J/Kg"),
            ("Qdot", "W"),
            ("Qdot_norm", "W"),
        ]
        columns = pandas.MultiIndex.from_tuples(variables)
        return pandas.DataFrame(columns=columns)

    def _extract_data(self, lowpass=False):
        """
        Extract relivant data variables and calculate Qdot.

        Parameters
        ----------
        lowpass : bool, optional
            Apply lowpass filter.
            Window length set equal to 2.5*period of driving waveform.
            The default is False.

        Returns
        -------
        data : pandas.DataFrame
            ACloss dataframe.

        """
        data = self._initialize_dataframe()
        data["t"] = self.sultandata["Time"]
        data["mdot"] = self.sultandata[f"dm/dt {self.side}"] * 1e-3
        data["Ipulse"] = self.sultandata["PS EEI (I)"]
        side = ["Left", "Right"]
        for end_index, end in enumerate(["in", "out"]):
            temperature_index = 1 + side.index(self.side) + 2 * end_index
            temperature_array = [
                f"T{temperature_index}{1 + probe}" for probe in range(4)
            ]
            data[f"T{end}"] = self.sultandata[temperature_array].mean(axis=1)
            # data[f'T{end}'] = self.sultandata[f'T {end} {self.side}']
            data[f"P{end}"] = self.sultandata[f"P {end} {self.side}"] * 1e5
        if lowpass:
            timestep = np.diff(data["t"], axis=0).mean()
            windowlength = int(0.5 / (timestep * self.frequency))
            if windowlength % 2 == 0:
                windowlength += 1
            if windowlength < 5:
                windowlength = 5
            for attribute in ["mdot", "Ipulse", "Tin", "Tout", "Pin", "Pout"]:
                data[attribute] = scipy.signal.savgol_filter(
                    np.squeeze(data[attribute]), windowlength, polyorder=3
                )
        for end in ["in", "out"]:  # Calculate enthapy
            temperature = data[f"T{end}"].values
            pressure = data[f"P{end}"].values
            data[f"h{end}"] = CoolProp.PropsSI(
                "H", "T", temperature, "P", pressure, "Helium"
            )
        # net heating
        data["Qdot"] = data[("mdot", "kg/s")] * (
            data[("hout", "J/Kg")] - data[("hin", "J/Kg")]
        )
        # per active length
        data["Qdot"] /= 0.39
        # normalize Qdot heating by |Bdot|**2
        data["Qdot_norm"] = data["Qdot"] / self.excitation_field_rate**2
        return data

    @property
    def raw(self):
        """Return raw data, read-only."""
        self._reload()
        if self.reload.raw:
            self._raw = self._extract_data(lowpass=False)
            self.reload.raw = False
        return self._raw

    @property
    def lowpass(self):
        """Return lowpass data, read-only."""
        self._reload()
        if self.reload.lowpass:
            self._lowpass = self._extract_data(lowpass=True)
            self.reload.lowpass = False
        return self._lowpass

    @property
    def data(self):
        """Return sample data corresponding to lowpass filter setting."""
        if self.lowpass_filter:
            data = self.lowpass
        else:
            data = self.raw
        return data

    @property
    def heatindex(self):
        """Return heatindex."""
        self._reload()
        if self.reload.heatindex:
            self._heatindex = HeatIndex(self.raw)
            self.reload.heatindex = False
        return self._heatindex


if __name__ == "__main__":
    trial = Trial("CSJA13", -1, "ac")
    sourcedata = SourceData(trial, 2)
    sampledata = SampleData(sourcedata, True)
