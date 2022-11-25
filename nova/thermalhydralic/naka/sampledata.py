"""Postprocess naka shotdata."""
import os
from dataclasses import dataclass, field
from typing import Union

import pandas
import numpy as np
import scipy

from nova.thermalhydralic.naka.database import DataBase
from nova.thermalhydralic.naka.nakadata import NakaData
from nova.utilities.pandasdata import PandasHDF
from nova.plot import plt


@dataclass
class SampleData(PandasHDF):
    """Post process Naka data."""

    nakadata: Union[NakaData, DataBase, int]
    _shot: int
    data: pandas.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        """Init nakadata instance, load postprocess chain."""
        if not isinstance(self.nakadata, NakaData):
            self.nakadata = NakaData(self.nakadata)
        self.shot = self._shot

    @property
    def shot(self):
        """Manage shot index."""
        return self._shot

    @shot.setter
    def shot(self, shot):
        self.nakadata.shot = shot
        self._shot = self.nakadata.shot
        #load data

    @property
    def database(self):
        """Return database instance."""
        return self.nakadata.database

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return os.path.join(self.database.binary_filepath('sampledata.h5'))

    @property
    def filename(self):
        """Manage datafile filename."""
        return self.database.shot_list[self.shot]

    def _read_data(self):
        columns = []
        for label in ['ICS_FL', 'ICS_TS', 'ICS_TC', 'ICS_PT', 'FAC_IC_CT',
                      'ICS_FCT']:
            columns.extend(self.nakadata.columns(label))
        nakadata = self.nakadata.data.loc[:, columns]
        labels = {column: '_'.join(column.split('_')[1:])
                  for column in columns}
        nakadata.rename(columns=labels, inplace=True)
        thermometers = [label for label in nakadata.columns
                        if label[0] == 'T' and 'STR' not in label]
        position = {
            'IN': -2.25 - 0.9285, '08': -2.25 - (0.9285-0.646),
            '07': -2.25, '06': -1.5, '05': -0.75, '04': 0.75, '03': 1.5,
            '02': 2.25, '01': 2.25 + (0.9285-0.646), 'OUT': 2.25 + 0.9285}

        pressure = scipy.interpolate.interp1d(
            np.array([position['IN'], position['OUT']]),
            nakadata.loc[:, ['PT_IN', 'PT_OUT']].to_numpy())
        massflow = scipy.interpolate.interp1d(
            np.array([position['IN'], position['OUT']]),
            nakadata.loc[:, ['FCT_INc', 'FCT_OUTc']].to_numpy())
        data = {'time': np.mean(self.nakadata.data.time, axis=1)}
        for label in position:
            data[f'P_{label.lower()}'] = pressure(position[label])
            data[f'mdot_{label.lower()}'] = massflow(position[label])
            thermometer_label = next(meter for meter in thermometers
                                     if label in meter)
            data[thermometer_label.replace(label, label.lower())] = \
                nakadata.loc[:, thermometer_label]

        data = pandas.DataFrame(data)
        data.attrs['position'] = position
        print(data)

        print(nakadata.columns.to_list())

        plt.plot(data.time, data.mdot_in)
        plt.plot(data.time, data.mdot_out)


if __name__ == '__main__':

    sample = SampleData(2015, 13)
    #sample.nakadata.read_data()

    sample._read_data()
    #nakadata._read_data()

    #plt.plot(sample.nakadata.data.time, sample.nakadata.data['ICS_VD_HR-IN'])
    #plt.plot(sample.nakadata.data.time, sample.nakadata.data['FAC_IC_CT1'])
    #plt.plot(sample.nakadata.data.time, sample.nakadata.data['FAC_IC_CT2'])


    '''
    #Tin = dataframe['ICS_TS_07L']
    #Tout = dataframe['ICS_TS_02L']

    #dT = Tout-Tin
    index = (dataframe.time > 1600) & (dataframe.time < 2000)
    dataframe = dataframe.loc[index, :]

    dp = dataframe['ICS_PT_IN']#-dataframe['ICS_PT_IN']
    #plt.plot(dataframe.time, dataframe['ICS_PT_IN'])
    plt.plot(dataframe.time, dp)
    #plt.xlim([0, 500])

    fs = 1/np.mean(np.diff(dataframe.time))
    f, Pxx = scipy.signal.welch(dp, fs)

    plt.figure()
    plt.loglog(f[1:], Pxx[1:])

    1/0.005
    '''


