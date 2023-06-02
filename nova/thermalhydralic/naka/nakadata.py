"""Manage local and remote naka test data."""
import os
from dataclasses import dataclass, field
from typing import Union

import pandas
import numpy as np

from nova.thermalhydralic.naka.database import DataBase
from nova.utilities.pandasdata import PandasHDF


@dataclass
class NakaData(PandasHDF):
    """Manage naka data."""

    database: Union[DataBase, str]
    _shot: int = 0
    data: pandas.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        """Load dataset."""
        if not isinstance(self.database, DataBase):
            self.database = DataBase(self.database)
        self.shot = self._shot

    @property
    def shot(self):
        """Manage shot index."""
        return self._shot

    @shot.setter
    def shot(self, shot):
        if shot < 0:
            shot = range(self.database.shot_number)[shot]
        self._shot = shot
        self.data = self.load_data()

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return os.path.join(self.database.binary_filepath("testdata.h5"))

    @property
    def filename(self):
        """Manage datafile filename."""
        return self.database.shot_list[self.shot]

    def _read_dataframe(self, file):
        """Return csv file data as pandas.DataFrame."""
        dataframe = pandas.read_csv(
            file, skiprows=7, dtype=float, na_values=["1.#INF00e+000"]
        )
        columns = {}
        for name in dataframe.columns:
            columns[name] = name.replace("(sec)", "")
            columns[name] = columns[name].replace("phy(", "").replace(")", "")
        dataframe.rename(columns=columns, inplace=True)
        dataframe.dropna(inplace=True, axis=1)
        return dataframe

    def _read_dataset(self, files):
        """Return dataset."""
        file_number = len(files)
        dataset = [[] for __ in range(file_number)]
        sample_number = np.zeros(file_number, dtype=int)
        for i, file in enumerate(files):
            dataset[i] = self._read_dataframe(file)
            sample_number[i] = dataset[i].shape[0]
        stop_index = np.min(sample_number)
        for i in range(file_number):
            dataset[i] = dataset[i].iloc[slice(stop_index), :]
            drop = dataset[i].select_dtypes(exclude=float).columns
            dataset[i].drop(columns=drop, inplace=True)
        dataset = pandas.concat(dataset, axis=1)
        return dataset

    def _read_data(self):
        """Return concatinated dataset."""
        files = self.database.locate(self.shot, files="L.csv")
        self.database.download()
        return self._read_dataset(files)

    def columns(self, label):
        """Return list of columns containing label."""
        return [column for column in self.data.columns if label in column]


if __name__ == "__main__":
    nakadata = NakaData(2015, 209)
    # nakadata.read_data()
