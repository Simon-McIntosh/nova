
"""Manage fast access dataframe attributes."""
from contextlib import contextmanager

from typing import Optional, Collection, Any

import pandas

from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.dataframe import DataFrame

# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object
# pylint: disable=protected-access


class DataFrameArray(DataFrame):
    """
    Extends DataFrame enabling fast access to dynamic fields in array.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns)
        self.metaarray.metadata = metadata

    @property
    def metaarray(self):
        """
        Return metaarray instance, protect against pandas recursion loop.

        To understand recursion, you must understand recursion.
        """
        self.update_metaarray()
        return self.attrs['metaarray']

    def update_metaarray(self):
        """Update metaarray if not present in self.attrs."""
        if 'metaarray' not in self.attrs:
            self.attrs['metaarray'] = MetaArray()

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        self.reload_frame()
        print('reload')
        return super().__repr__()

    def reload_frame(self):
        """Transfer data from metaarray.data to frame."""
        for col in self.metaarray.array:
            self._update_frame(col)

    def _update_array(self, index=None, col=None, value=None):
        if index is None:
            index = slice(None)
        elif isinstance(index, str):
            _index = self.index.get_indexer(index)
            if _index == -1:
                raise IndexError(f'index {index} not found in {self.index}')
            index = _index
        if value is None:
            value = pandas.DataFrame.__getattr__(self, col).to_numpy()
        with self._setarray(col):
            self.metaarray.data[col][index] = value

    @contextmanager
    def _setarray(self, col):
        if not hasattr(self.metaarray.data, col):
            self.metaarray.data[col] = \
                pandas.DataFrame.__getattr__(self, col).to_numpy()
        yield
        self.metaarray.update_array[col] = False

    def _update_frame(self, col):
        if self.metaarray.update_frame[col]:
            with self._setframe(col):
                super().__setitem__(col, self.metaarray.data[col])

    @contextmanager
    def _setframe(self, col):
        yield
        self.metaarray.update_frame[col] = False

    def __getattr__(self, col):
        """Extend DataFrame.__getattr__. (frame.*)."""
        if col in self.metaarray.array:
            if self.metaarray.update_array[col]:
                self._update_array(col=col)
            return self.metaarray.data[col]
        return super().__getattr__(col)

    def __setattr__(self, col, value):
        """Extend DataFrame.__setattr__ (frame.* = *).."""
        if col in self.metaarray.array:
            self._update_array(col=col, value=value)
            self.metaarray.update_frame[col] = True
            return None
        return super().__setattr__(col, value)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if col in self.metaarray.array:
            self._update_frame(col)
        return super().__getitem__(col)

    def _get_value(self, index, col, takeable=False):
        """Extend DataFrame._get_value. (frame.at[i, '*'])."""
        if col in self.metaarray.array:
            self._update_frame(col)
        return super()._get_value(index, col, takeable)

    def __setitem__(self, col, value):
        """Extend DataFrame.__setitem__. (frame['*'] = *)."""
        if col in self.metaarray.array:
            self._update_array(col=col, value=value)
        super().__setitem__(col, value)

    def _set_value(self, index, col, value, takeable=False):
        """Extend DataFrame._set_value. (frame.at[i, '*'] = *)."""
        if col in self.metaarray.array:
            self._update_array(index=index, col=col, value=value)
        return super()._set_value(index, col, value, takeable)


if __name__ == '__main__':

    dataframearray = DataFrameArray()
