"""Postprocess Sultan test data."""
from dataclasses import dataclass, field
from typing import Optional
from types import SimpleNamespace

import pandas

from nova.thermalhydralic.sultan.database import DataBase
from nova.utilities.pandasdata import PandasHDF


@dataclass
class SultanData(PandasHDF):
    """Manage Sultan timeseries data."""

    database: DataBase
    _filename: Optional[str] = None
    _data: pandas.DataFrame | None = field(init=False, repr=False, default=None)
    reload: SimpleNamespace = field(
        init=False, repr=False, default_factory=SimpleNamespace
    )

    def __post_init__(self):
        """Typecheck database."""
        self.reload.__init__(data=True)
        if not isinstance(self.database, DataBase):
            self.database = DataBase(self.database)
        if self._filename is not None:
            self.filename = self._filename

    @property
    def data(self):
        """Return sultandata."""
        if self.reload.data:
            self.data = self.load_data()
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.reload.data = False

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.database.binary_filepath("testdata.h5")

    @property
    def filepath(self):
        """Return full path of source datafile, read-only."""
        return self.database.datafile(self.filename)

    @property
    def filename(self):
        """Manage datafile filename."""
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename
        self.reload.data = True

    def _read_data(self):
        """
        Return sultan dataframe.

        Returns
        -------
        data : pandas.DataFrame
            Shot data.

        """
        data = pandas.read_csv(self.filepath, encoding="ISO-8859-1")
        columns = {}
        for column in data.columns:
            if "left" in column or "right" in column:
                columns[column] = column.replace("left", "Left")
                columns[column] = columns[column].replace("right", "Right")
                columns[column] = columns[column].replace("  ", " ")
            if column[-7:] == " (320K)":
                columns[column] = column[:-7]
        data.rename(columns=columns, inplace=True)
        if "T in" in data.columns:
            data["T in Left"] = data["T in"]
            data["T in Right"] = data["T in"]
        if "P in" in data.columns:
            data["P in Left"] = data["P in"]
            data["P in Right"] = data["P in"]
        return data
