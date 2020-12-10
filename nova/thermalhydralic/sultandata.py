"""Postprocess Sultan AC loss test data."""
from dataclasses import dataclass, field

import pandas

from nova.thermalhydralic.Sultan.sultanshot import SultanShot


@dataclass
class SultanData:
    """Manage Sultan timeseries data."""

    shot: SultanShot
    _shotindex: int = 0
    binary: bool = True
    _raw: pandas.DataFrame = field(init=False, repr=False, default=None)
    _lowpass: pandas.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Typecheck shot."""
        if isinstance(self.shot, str):
            self.shot = SultanShot(self.shot)

    @property
    def database(self):
        """Return database instance."""
        return self.shot.database

    @property
    def testplan(self):
        """Return testplan instance."""
        return self.shot.testplan

    @property
    def binaryfile(self):
        """Return full path of binary datafile."""
        return self.database.binary_filepath('testdata.h5')

    def load_datafile(self):
        """Return datafile."""
        rawdata = self._read_datafile()
        self._save_datafile('rawdata', rawdata)
        datafile = self._load_datafile('rawdata')
        print(datafile)

    def _load_datafile(self, key):
        """Return datafile from binary store."""
        with pandas.HDFStore(self.binaryfile, mode='r') as store:
            datafile = store[key]
        return datafile

    def _read_datafile(self):
        """
        Return sultan dataframe.

        Returns
        -------
        sultandata : pandas.DataFrame
            Shot data.

        """
        print(self.test.datafile)
        sultandata = pandas.read_csv(self.test.datafile, encoding='ISO-8859-1')
        columns = {}
        for column in sultandata.columns:
            if 'left' in column or 'right' in column:
                columns[column] = column.replace('left', 'Left')
                columns[column] = columns[column].replace('right', 'Right')
                columns[column] = columns[column].replace('  ', ' ')
            if column[-7:] == ' (320K)':
                columns[column] = column[:-7]
        sultandata.rename(columns=columns, inplace=True)
        if 'T in' in sultandata.columns:
            sultandata['T in Left'] = sultandata['T in']
            sultandata['T in Right'] = sultandata['T in']
        if 'P in' in sultandata.columns:
            sultandata['P in Left'] = sultandata['P in']
            sultandata['P in Right'] = sultandata['P in']
        return sultandata

    def _save_datafile(self, key, dataframe):
        """Append dataframe to hdf file."""
        with pandas.HDFStore(self.binaryfile, mode='w') as store:
            store.put(key, dataframe, format='table', append=True)


if __name__ == '__main__':

    data = SultanData('CSJA_3')
