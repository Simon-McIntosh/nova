"""Postprocess Sultan AC loss test data."""
from dataclasses import dataclass, field
from typing import Optional

import pandas

from nova.thermalhydralic.sultan.database import DataBase
from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.thermalhydralic.sultan.shotinstance import ShotInstance


@dataclass
class SultanData:
    """Manage Sultan timeseries data."""

    database: DataBase
    _filename: Optional[str] = None
    data: pandas.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Typecheck database."""
        if not isinstance(self.database, DataBase):
            self.database = DataBase(self.database)
        if self._filename is not None:
            self.filename = self._filename

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.database.binary_filepath('testdata.h5')

    @property
    def filepath(self):
        """Return full path of source datafile, read-only."""
        return self.database.source_filepath(self.filename) + '.dat'

    @property
    def filename(self):
        """Manage datafile filename."""
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename
        self.load_data()

    def load_data(self):
        """Return raw sultan data."""
        try:
            data = self._load_data()
        except (KeyError, OSError):
            data = self._read_data()
            self._save_data(data)
        self.data = data

    def _load_data(self):
        """Return data from binary store."""
        with pandas.HDFStore(self.binaryfilepath, mode='r') as store:
            data = store[self.filename]
        return data

    def _read_data(self):
        """
        Return sultan dataframe.

        Returns
        -------
        data : pandas.DataFrame
            Shot data.

        """
        data = pandas.read_csv(self.filepath, encoding='ISO-8859-1')
        columns = {}
        for column in data.columns:
            if 'left' in column or 'right' in column:
                columns[column] = column.replace('left', 'Left')
                columns[column] = columns[column].replace('right', 'Right')
                columns[column] = columns[column].replace('  ', ' ')
            if column[-7:] == ' (320K)':
                columns[column] = column[:-7]
        data.rename(columns=columns, inplace=True)
        if 'T in' in data.columns:
            data['T in Left'] = data['T in']
            data['T in Right'] = data['T in']
        if 'P in' in data.columns:
            data['P in Left'] = data['P in']
            data['P in Right'] = data['P in']
        return data

    def _save_data(self, dataframe):
        """Append dataframe to hdf file."""
        with pandas.HDFStore(self.binaryfilepath, mode='w') as store:
            store.put(self.filename, dataframe, format='table', append=True)


if __name__ == '__main__':

    testplan = TestPlan('CSJA_3')
    shotinstance = ShotInstance(testplan)

    sultandata = SultanData(testplan.database, shotinstance.filename)
