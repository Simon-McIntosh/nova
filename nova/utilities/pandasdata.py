"""Abstract methods for managing pandas dataframes."""

import abc

import pandas


class PandasHDF(metaclass=abc.ABCMeta):
    """Manage dataframe input/output."""

    def load_data(self):
        """Load data from HDF store."""
        try:
            data = self._load_data()
        except (KeyError, OSError):
            data = self.read_data()
        return data

    def read_data(self):
        """Return raw data from file."""
        data = self._read_data()
        self._save_data(data)
        return data

    def _load_data(self):
        """Return data from binary store."""
        with pandas.HDFStore(self.binaryfilepath, mode='r') as store:
            data = store[self.filename]
            if 'metadata' in store.get_storer(self.filename).attrs:
                data.attrs = store.get_storer(self.filename).attrs.metadata
        return data

    def _save_data(self, data):
        """Append data to hdf file."""
        with pandas.HDFStore(self.binaryfilepath, mode='a') as store:
            store.put(self.filename, data, format='table', append=False)
            if data.attrs:
                store.get_storer(self.filename).attrs.metadata = data.attrs

    @property
    @abc.abstractmethod
    def binaryfilepath(self) -> str:
        """Return full filepath for binary file."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        """Return data filename."""
        raise NotImplementedError

    @abc.abstractmethod
    def _read_data(self):
        """Return dataframe."""
        raise NotImplementedError
