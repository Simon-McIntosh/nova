"""Manage fast access dataframe attributes."""
from contextlib import contextmanager

from typing import Optional, Collection, Any

import pandas

from nova.electromagnetic.metaarray import MetaArray

# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object


class Series(pandas.Series):
    """Provide series constructor methods."""

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        return DataFrame


class DataFrame(pandas.DataFrame):
    """pandas.DataFrame base class."""

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series


class SeriesArray(pandas.Series):
    """Provide series constructor methods."""

    @property
    def _constructor(self):
        return SeriesArray

    @property
    def _constructor_expanddim(self):
        return DataFrameArray


class DataFrameArray(pandas.DataFrame):
    """
    Extends Pandas.DataFrame enabling fast access to dynamic fields in array.

    Fast access variables stored in ...
    Lazy data exchange implemented with parent DataFrame.

    Inspiration for DataFrame inheritance taken from GeoPandas
    https://github.com/geopandas.

    Extended by Frame. Inherited alongside DataFrame.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None):
        super().__init__(data, index, columns)
        if isinstance(data, pandas.core.internals.managers.BlockManager):
            return
        self.update_metaarray()

    @property
    def _constructor(self):
        return DataFrameArray

    @property
    def _constructor_sliced(self):
        return SeriesArray

    def update_metaarray(self):
        """Update metaarray if not present in self.attrs."""
        if 'metaarray' not in self.attrs:
            self.attrs['metaarray'] = MetaArray()

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        self.reload_frame()
        return pandas.DataFrame.__repr__(self)

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
                pandas.DataFrame.__setitem__(self, col,
                                             self.metaarray.data[col])

    @contextmanager
    def _setframe(self, col):
        yield
        self.metaarray.update_frame[col] = False

    @property
    def metaarray(self):
        """
        Return metaarray instance, protect against pandas recursion loop.

        To understand recursion, you must understand recursion.
        """
        return self.attrs['metaarray']

    def __getattr__(self, col):
        """Extend pandas.DataFrame.__getattr__. (frame.*)."""
        if col in self.metaarray.array:
            if self.metaarray.update_array[col]:
                self._update_array(col=col)
            return self.metaarray.data[col]
        return pandas.DataFrame.__getattr__(self, col)

    def __setattr__(self, col, value):
        """Extend pandas.DataFrame.__setattr__ (frame.* = *).."""
        if col in self.metaarray.array:
            self._update_array(col=col, value=value)
            self.metaarray.update_frame[col] = True
            return None
        return pandas.DataFrame.__setattr__(self, col, value)

    def __getitem__(self, col):
        """Extend pandas.DataFrame.__getitem__. (frame['*'])."""
        if col in self.metaarray.array:
            self._update_frame(col)
        return pandas.DataFrame.__getitem__(self, col)

    def _get_value(self, index, col, takeable=False):
        """Extend pandas.DataFrame._get_value. (frame.at[i, '*'])."""
        if col in self.metaarray.array:
            self._update_frame(col)
        return pandas.DataFrame._get_value(  # pylint: disable=protected-access
                                           self, index, col, takeable)

    def __setitem__(self, col, value):
        """Extend pandas.DataFrame.__setitem__. (frame['*'] = *)."""
        if col in self.metaarray.array:
            self._update_array(col=col, value=value)
        pandas.DataFrame.__setitem__(self, col, value)

    def _set_value(self, index, col, value, takeable=False):
        """Extend pandas.DataFrame._set_value. (frame.at[i, '*'] = *)."""
        if col in self.metaarray.array:
            self._update_array(index=index, col=col, value=value)
        return pandas.DataFrame._set_value(  # pylint: disable=protected-access
                                           self, index, col, value, takeable)



if __name__ == '__main__':

    dataframe = DataFrame()
