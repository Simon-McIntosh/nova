"""Manage fast access dataframe attributes."""
from typing import Optional, Collection, Any

import numpy as np
import pandas

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.indexer import Indexer

# pylint: disable=too-many-ancestors


class ArrayLocMixin():
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Extend Loc setitem."""
        col = self.obj.get_col(key)
        if self.obj.metaframe.hascol('array', col):
            index = self.obj.get_index(key)
            if isinstance(index, slice):
                if index == slice(None):
                    return self.obj.__setitem__(col, value)
            try:
                self.obj.__getitem__(col)[index] = value
                return None
            except KeyError:
                pass
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Extend Loc getitem. Update frame prior to return if col in array."""
        col = self.obj.get_col(key)
        if self.obj.metaframe.hascol('array', col):
            if self.obj.metaframe.lock('array') is False:
                self.obj._set_frame(col)  # update frame
                with self.obj.metaframe.setlock(True, 'array'):
                    return super().__getitem__(key)
        return super().__getitem__(key)


class ArrayIndexer(Indexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return ArrayLocMixin


class DataArray(ArrayIndexer, DataFrame):
    """Extend DataFrame enabling fast access to dynamic fields in array."""

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Collection[Any]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)

    def __repr__(self):
        """Propagate array variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def update_frame(self):
        """Extend DataFrame.update_frame, transfer metaarray.data to frame."""
        for col in self.metaframe.array:
            self._set_frame(col)

    def __setattr__(self, col, value):
        """Extend DataFrame.__setattr__ to gain fast access to array data."""
        if self.metaframe.hascol('array', col):
            return self._set_array(col, value)
        return super().__setattr__(col, value)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.metaframe.hascol('array', col):
            if self.metaframe.lock('array') is False:
                return self._get_array(col)
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Extend DataFrame.__setitem__. (frame['*'] = *)."""
        if self.metaframe.hascol('array', col):
            if self.metaframe.lock('array') is False:
                return self._set_array(col, value)
        return super().__setitem__(col, value)

    def _get_array(self, col):
        """Return col, quickly."""
        try:
            return self.attrs['metaframe'].data[col]
        except KeyError:
            value = self._get_frame(col)
            self.attrs['metaframe'].data[col] = value
            return value

    def _set_array(self, col, value):
        """Set col, quickly, preserving shape."""
        try:
            self.attrs['metaframe'].data[col][:] = value
        except KeyError:
            if not pandas.api.types.is_list_like(value):
                value = np.full(len(self), value)
            elif len(value) != len(self):
                raise IndexError(f'input length {len(value)} != {len(self)}')
            self.attrs['metaframe'].data[col] = value

    def _get_frame(self, col):
        """Return col from frame."""
        try:
            value = super().__getitem__(col).to_numpy()
        except (AttributeError, KeyError):
            value = np.full(len(self), self.metaframe.default[col])
            super().__setitem__(col, value)
        return value

    def _set_frame(self, col):
        """Transfer metaarray.data to frame."""
        self.metaframe.assert_hascol('array', col)
        value = self._get_array(col)
        with self.metaframe.setlock(True, 'array'):
            super().__setitem__(col, value)


if __name__ == '__main__':

    dataarray = DataArray({'x': range(7), 'z': 0},
                          additional=['Ic'], Array=['Ic'], label='Coil')

    dataarray = DataArray({'x': range(7)},
                          additional=['Ic'], Array=['x'], label='Coil')
    print(dataarray)

    '''
    dataarray.Ic = 9

    dataarray = DataArray({'x': range(12)}, link=True,
                          Required=['x'], Array=['x'])

    dataarray.loc['Coil0':'Coil1', 'x'] = 6.66
    print(dataarray)
    '''
