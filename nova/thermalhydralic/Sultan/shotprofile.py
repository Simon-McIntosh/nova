"""Manage sultan timeseries data."""
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Union

import pandas
import numpy as np

from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.utilities.pyplot import plt


@dataclass
class PointData:
    """Extract point data from ShotData."""

    data: pandas.DataFrame = field(repr=False)
    start_index: int
    _offset: Union[bool, tuple[float]] = True
    normalize: bool = True
    time_label: tuple[str] = ('t', 's')
    _data_label: tuple[str] = ('Qdot', 'W')

    def __post_init__(self):
        """Calculate offset."""
        self.offset = self._offset

    @property
    def data_label(self):
        """Return data label."""
        if self.normalize:
            label = (self._data_label[0]+'_norm', self._data_label[1])
        else:
            label = self._data_label
        return label

    @property
    def offset(self):
        """
        Manage data offset.

        Parameters
        ----------
        offset : bool or tuple[float, float]

            - True: offset data to heatinstance.start
            - False: no offset
            - tuple: custom offset

        Returns
        -------
        offset: tuple[float, float]
            offset.

        """
        return self._offset

    @offset.setter
    def offset(self, offset):
        if isinstance(offset, bool):
            self._offset = (0, 0)
            if offset:
                self._offset = self.point(self.start_index)
        else:
            self._offset = offset

    def _loc(self, prefix, index):
        variables = ['time', 'data']
        if prefix not in variables:
            raise NameError(f'prefix {prefix} not in [time, data]')
        label = getattr(self, f'{prefix}_label')
        offset_index = variables.index(prefix)
        return self.data.loc[index, label] - self.offset[offset_index]

    def point(self, index):
        """Return time, data tuple."""
        return self._loc('time', index), self._loc('data', index)

    def plot(self, index, *args, **kwargs):
        """
        Plot point data.

        Parameters
        ----------
        index : int
            Data index.
        **kwargs : any
            plot keyword args.

        Returns
        -------
        None.

        """
        axes = kwargs.pop('axes', None)
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        axes.plot(*self.point(index), *args, **kwargs)


@dataclass
class ShotProfile:
    """Extract and filter sultan timeseries data."""

    instance: Union[ShotInstance, TestPlan, str]
    _offset: Union[bool, tuple[float]] = True
    _normalize: bool = True
    _pointdata: PointData = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Build data pipeline."""
        self.reload.__init__(pointdata=True, response=True, waveform=True)

        if not isinstance(self.instance, ShotInstance):
            self.instance = ShotInstance(self.instance)
        self.offset = self._offset
        self.normalize = self._normalize

    @property
    def offset(self):
        """Manage data offset flag."""
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset
        self.shotdata.offset = offset

    @property
    def normalize(self):
        """Manage normalization flag."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        self._normalize = normalize
        self.shotdata.normalize = normalize

    @property
    def pointdata(self):
        """Return pointdata, read-only."""
        self._reload()
        if self.reload.pointdata:
            self._pointdata = PointData(self.lowpassdata, self.heatindex.start,
                                        self.offset, self.normalize)
            self.reload.pointdata = False
        return self._pointdata

    @property
    def side(self):
        """Return experiment side, read-only."""
        return self.instance.side

    @property
    def filename(self):
        """Return instance filename."""
        return self.instance.filename

    @property
    def shotname(self):
        """Return instance shotname."""
        return self.instance.shotname

    @property
    def frequency(self):
        """Return instance frequency, Hz."""
        return self.instance.frequency

    @property
    def omega(self):
        """Return instance frequency, rad/s."""
        return 2*np.pi*self.instance.frequency

    def _reload(self):
        """Set data chain reload flags."""
        if self.instance.reload.sultandata:
            self.reload.response = True
            self.reload.waveform = True



    def plot_single(self, variable, axes=None, lowpass=False):
        """
        Plot single waveform.

        Parameters
        ----------
        variable : str
            variable name.
        axes : axes, optional
            Plot axes. The default is None, plt.gca().
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
        if axes is None:
            axes = plt.gca()
        bg_color = 0.4 * np.ones(3) if lowpass else 'lightgray'
        color = 'C3' if lowpass else 'C0'
        label = 'lowpass' if lowpass else 'raw'
        axes.plot(data.t-self.shotdata.offset[0],
                  data[variable]-self.shotdata.offset[1],
                  color=bg_color)
        axes.plot(
            data.t[self.heatindex.index]-self.shotdata.offset[0],
            data[variable][self.heatindex.index]-self.shotdata.offset[1],
            color=color, label=label)
        axes.legend()
        axes.set_xlabel('$t$ s')
        axes.set_ylabel(r'$\hat{\dot{Q}}$ W')
        plt.despine()

    def plot(self, axes=None):
        """Plot shot profile."""
        self.plot_single('Qdot_norm', lowpass=False, axes=axes)
        self.plot_single('Qdot_norm', lowpass=True, axes=axes)


if __name__ == '__main__':

    shotprofile = ShotProfile('CSJA13')
    shotprofile.instance.index = -5
    shotprofile.plot()
