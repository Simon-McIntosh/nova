"""Manage sultan shot data."""
from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas
import numpy as np
import scipy.signal
import CoolProp.CoolProp as CoolProp

from nova.thermalhydralic.sultandata import SultanData
from nova.thermalhydralic.heatindex import HeatIndex


@dataclass
class ShotData:
    """Manage shot dataframes."""

    sultan: SultanData
    _rawdata: pandas.DataFrame = field(init=False, repr=False)
    _lowpassdata: pandas.DataFrame = field(init=False, repr=False)
    _heatindex: HeatIndex = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init data pipeline."""
        self.reload.__init__(rawdata=True, lowpassdata=True, heatindex=True)

    @staticmethod
    def _initialize_dataframe():
        """
        Return calclation dataframe.

        Returns
        -------
        dataframe : pandas.DataFrame
            Empty dataframe with time index and default columns names.

        """
        variables = [('t', 's'), ('mdot', 'kg/s'), ('Ipulse', 'A'),
                     ('Tin', 'K'), ('Tout', 'K'),
                     ('Pin', 'Pa'), ('Pout', 'Pa'),
                     ('hin', 'J/Kg'), ('hout', 'J/Kg'),
                     ('Qdot', 'W'), ('Qdot_norm', 'W')]
        columns = pandas.MultiIndex.from_tuples(variables)
        return pandas.DataFrame(columns=columns)

    def _extract_data(self, lowpass=False):
        """
        Extract relivant data variables and calculate Qdot.

        Parameters
        ----------
        sultandata : pandas.DataFrame
            Sultan shot data.
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
        data['t'] = self.sultandata['Time']
        data['mdot'] = self.sultandata[f'dm/dt {self.side}'] * 1e-3
        data['Ipulse'] = self.sultandata['PS EEI (I)']
        for end in ['in', 'out']:
            data[f'T{end}'] = self.sultandata[f'T {end} {self.side}']
            data[f'P{end}'] = self.sultandata[f'P {end} {self.side}'] * 1e5
        if lowpass:
            timestep = np.diff(data['t'], axis=0).mean()
            windowlength = int(2.5 / (timestep*self.frequency))
            if windowlength % 2 == 0:
                windowlength += 1
            if windowlength < 5:
                windowlength = 5
            for attribute in ['mdot', 'Ipulse', 'Tin', 'Tout', 'Pin', 'Pout']:
                data[attribute] = scipy.signal.savgol_filter(
                    np.squeeze(data[attribute]), windowlength, polyorder=3)
        for end in ['in', 'out']:  # Calculate enthapy
            temperature = data[f'T{end}'].values
            pressure = data[f'P{end}'].values
            data[f'h{end}'] = CoolProp.PropsSI('H', 'T', temperature,
                                               'P', pressure, 'Helium')
        # net heating
        data['Qdot'] = data[('mdot', 'kg/s')] * \
            (data[('hout', 'J/Kg')] - data[('hin', 'J/Kg')])
        # normalize Qdot heating by |Bdot|**2
        data['Qdot_norm'] = data['Qdot'] / self.excitation_field_rate**2
        return data

    @property
    def excitation_field(self):
        """
        Return amplitude of excitation field.

        Parameters
        ----------
        Ipulse : str
            Sultan Ipulse field.

        Returns
        -------
        excitation_field : float
            External excitation field.

        """
        try:
            current = float(re.findall(r'\d+', self.instance.current_label)[0])
        except TypeError:
            current = 230
        excitation_field = current * 0.2/230  # excitation field amplitude
        return excitation_field

    @property
    def excitation_field_rate(self):
        """Return amplitude of exciation field rate of change."""
        return self.omega*self.excitation_field  # pulse field rate amplitude

    @property
    def rawdata(self):
        """Return rawdata, read-only."""
        self._reload()
        if self.reload.rawdata:
            self._rawdata = self._extract_data(lowpass=False)
            self.reload.rawdata = False
        return self._rawdata

    @property
    def lowpassdata(self):
        """Return lowpassdata, read-only."""
        self._reload()
        if self.reload.lowpassdata:
            self._lowpassdata = self._extract_data(lowpass=True)
            self.reload.lowpassdata = False
        return self._lowpassdata

    @property
    def heatindex(self):
        """Return heatindex."""
        self._reload()
        if self.reload.heatindex:
            self._heatindex = HeatIndex(self.rawdata)
            self.reload.heatindex = False
        return self._heatindex
