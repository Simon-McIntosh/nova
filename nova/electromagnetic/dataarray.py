
"""Manage fast access dataframe attributes."""
from contextlib import contextmanager

from typing import Optional, Collection, Any

import pandas
import numpy as np

from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.dataframe import DataFrame, FrameMixin
from nova.electromagnetic.indexer import Indexer


# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object
# pylint: disable=protected-access

class ArrayMixin(FrameMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        print('set loc', key)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        print('get loc', key)
        return super().__getitem__(key)


class DataArray(DataFrame):
    """
    Extends DataFrame enabling fast access to dynamic fields in array.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Collection[Any]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)

    def _extract_attrs(self, data, attrs):
        """Extend DataFrame._extract_attrs, insert metaarray."""
        super()._extract_attrs(data, attrs)
        if not self._hasattr('metaarray'):
            self.attrs['metaarray'] = MetaArray()  # init metaframe
        else:
            print('post init')
            self.metaarray.__post_init__()
        self.attrs['indexer'] = Indexer(ArrayMixin)

    def update_frame(self):
        """Transfer data from metaarray.data to frame."""
        for col in self.metaarray.array:
            self._update_frame(col)
        super().update_frame()

    @profile
    def _update_array(self, index=None, col=None, value=None):
        if index is None:
            index = slice(None)
        elif isinstance(index, str):
            _index = self.index.get_indexer(index)
            if _index == -1:
                raise IndexError(f'index {index} not found in {self.index}')
            index = _index
        if value is None:
            value = self._getcol(col)
        self._setarray(index, col, value)

    @profile
    def _setarray(self, index, col, value):
        """Set value in metaarray.data."""
        if col not in self.metaarray.data or index == slice(None):
            self.metaarray.data[col] = value
        else:
            self.metaarray.data[col][index] = value
        if col not in self.columns:
            super().__setitem__(col, value)
        self.metaarray.update_array[col] = False

    @profile
    def _getcol(self, col):
        """Return col from frame."""
        try:
            value = super().__getitem__(col).to_numpy()
        except (AttributeError, KeyError):
            value = np.full(len(self), self.metaframe.default[col])
        return self._format_value(col, value)

    def _update_frame(self, col):
        """Copy col data to frame."""
        if self.metaarray.update_frame[col]:
            super().__setitem__(col, self[col])
            self.metaarray.update_frame[col] = False

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if col in self.metaarray.array:
            self.metaarray.update_frame[col] = True
            if self.metaarray.update_array[col]:
                self._update_array(col=col)
            return self.metaarray.data[col]
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Extend DataFrame.__setitem__. (frame['*'] = *)."""
        if col in self.metaarray.array:
            self.metaarray.update_frame[col] = True
            return self._update_array(col=col, value=value)
        return super().__setitem__(col, value)



if __name__ == '__main__':

    dataarray = DataArray({'x': range(7), 'z': 0},
                          additional=['Ic'], Array=['Ic'], label='Coil')
    dataarray.add_frame(1, range(3))
    dataarray.Ic = 7.77
    print(dataarray)

    '''
    also...
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=['x'])
    #dataarray.loc[:, 'x'] = 5
    print(dataarray.x)
    '''



