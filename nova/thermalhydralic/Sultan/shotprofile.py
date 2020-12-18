"""Manage sultan timeseries data."""
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas
import numpy as np
import scipy.signal
import CoolProp.CoolProp as CoolProp

from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.sultandata import SultanData
from nova.thermalhydralic.sultan.shotresponse import HeatIndex, ShotResponse
from nova.utilities.pyplot import plt


@dataclass
class ShotProfile:
    """Extract and filter sultan timeseries data."""

    shotinstance: ShotInstance
    _side: str = 'Left'
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)
    _rawdata: pandas.DataFrame = field(init=False, repr=False)
    _lowpassdata: pandas.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        """Build data pipeline."""
        self.reload.__init__(side=True, rawdata=True, lowpassdata=True)
        self._sultandata = SultanData(self.shotinstance.database)

    @property
    def sultandata(self):
        """Return sultan datafile, update shot filename if required."""
        if self.shotinstance.reload.data:
            self._sultandata.filename = self.shotinstance.filename
        return self._sultandata.data

    @property
    def side(self):
        """
        Manage side property. reload raw and lowpass data if changed.

        Parameters
        ----------
        side : str
            Side of Sultan experement ['Left', 'Right'].

        Returns
        -------
        side : str

        """
        if self.reload.side:
            self.side = self._side
        return self._side

    @side.setter
    def side(self, side):
        side = side.capitalize()
        if side not in ['Left', 'Right']:
            raise IndexError(f'side {side} not in [Left, Right]')
        self._side = side
        self.reload.side = False
        self.reload.rawdata = True
        self.reload.lowpassdata = True

    @property
    def rawdata(self):
        """Return rawdata, read-only."""
        if self.reload.rawdata:
            self._rawdata = self._extract_data(lowpass=False)
            self.reload.rawdata = False
        return self._rawdata

    @property
    def lowpassdata(self):
        """Return lowpassdata, read-only."""
        if self.reload.lowpassdata:
            self._lowpassdata = self._extract_data(lowpass=True)
            self.reload.lowpassdata = False
        return self._lowpassdata

    @property
    def shotresponse(self):
        """Return shot response, read-only."""
        return ShotResponse(self.lowpassdata, HeatIndex(self.rawdata))

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
            windowlength = int(2.5 / (timestep*self.shotinstance.frequency))
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
        data['Qdot_norm'] = data['Qdot'] / self.external_field_rate**2
        return data

    @property
    def external_field(self):
        """
        Return amplitude of excitation field.

        Parameters
        ----------
        Ipulse : str
            Sultan Ipulse field.

        Returns
        -------
        external_field : float
            External excitation field.

        """
        try:
            current = float(re.findall(r'\d+',
                                       self.shotinstance.current_label)[0])
        except TypeError:
            current = 230
        external_field = current * 0.2/230  # excitation field amplitude
        return external_field

    @property
    def external_field_rate(self):
        """Return amplitude of exciation field rate of change."""
        omega = 2*np.pi*self.shotinstance.frequency
        return omega*self.external_field  # pulse field rate amplitude

    def plot_single(self, variable, ax=None, lowpass=False):
        """
        Plot single waveform.

        Parameters
        ----------
        variable : str
            variable name.
        ax : axis, optional
            Plot axis. The default is None, plt.gca().
        lowpass : bool, optional
            Serve lowpass filtered data. The default is False.

        Raises
        ------
        IndexError
            variable not found in dataframe.

        Returns
        -------
        None.

        """
        data = self.lowpassdata if lowpass else self.rawdata
        heat_index = self.shotresponse.heat_index
        if variable not in data:
            raise IndexError(f'variable {variable} not in {data.columns}')
        if ax is None:
            ax = plt.gca()
        bg_color = 0.4 * np.ones(3) if lowpass else 'lightgray'
        color = 'C3' if lowpass else 'C0'
        label = 'lowpass' if lowpass else 'raw'
        ax.plot(data.t, data[variable], color=bg_color)
        ax.plot(data.t[heat_index.index], data[variable][heat_index.index],
                color=color, label=label)
        ax.legend()
        ax.set_xlabel('$t$ s')
        ax.set_ylabel(r'$\hat{\dot{Q}}$ W')
        plt.despine()

    def plot(self):
        """Plot shot profile."""
        self.plot_single('Qdot_norm', lowpass=False)
        self.plot_single('Qdot_norm', lowpass=True)


if __name__ == '__main__':

    shotinstance = ShotInstance('CSJA_3')
    shotprofile = ShotProfile(shotinstance)

    shotinstance.index = -3

    shotprofile.plot()
