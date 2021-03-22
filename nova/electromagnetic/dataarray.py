"""Manage fast access dataframe attributes."""
from typing import Optional, Collection, Any

import numpy as np

from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.indexer import Indexer

# pylint: disable=too-many-ancestors


class ArrayLocMixin():
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Extend Loc setitem."""
        col = self.obj.get_col(key)
        if col in self.obj.metaarray.array:
            index = self.obj.get_index(key)
            if key == slice(None):
                return self.obj.__setitem__(col, value)
            self.obj.__getitem__(col)[index] = value
            return None
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Extend Loc getitem. Update frame prior to return if col in array."""
        col = self.obj.get_col(key)
        if col in self.obj.metaarray.array:
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

    def __getitem__(self, key):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        col = self.get_col(key)
        if col in self.metaarray.array:
            if self.metaframe.lock('array') is False:
                return self._get_array(col)
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Extend DataFrame.__setitem__. (frame['*'] = *)."""
        try:
            return self._set_array(col, value)
        except KeyError:
            return super().__setitem__(col, value)

    def _set_array(self, col, value):
        """Set col, quickly, preserving shape."""
        self.attrs['metaarray'].data[col][:] = value

    def _set_frame(self, col):
        """Transfer metaarray.data to frame."""
        print('data to frame', col)
        self.assert_in_field(col, 'array')
        #with self.metaframe.setlock(None):
        value = self.__getitem__(col)
        with self.metaframe.setlock(True, 'array'):
            #print(value)
            super().__setitem__(col, value)

    def extract_attrs(self, data, attrs):
        """Extend DataFrame.extract_attrs, insert metaarray."""
        super().extract_attrs(data, attrs)
        if not self.hasattrs('metaarray'):
            self.attrs['metaarray'] = MetaArray(self.index)  # init metaframe
        else:
            self.metaarray.__post_init__(self.index)

    def extract_metadata(self, metadata):
        """Extend DataFrame.extract_metadata, init data structure."""
        super().extract_metadata(metadata)
        if not self.empty:
            self.update_array()

    def update_array(self):
        """Set array data and backpropagate to frame if unset (default)."""
        for col in self.metaarray.array:
            self.attrs['metaarray'].data[col] = self._get_frame(col)

    def update_frame(self):
        """Extend DataFrame.update_frame, transfer metaarray.data to frame."""
        super().update_frame()
        # TODO - fix frame update
        for col in self.metaarray.array:
            self._set_frame(col)


    def _get_array(self, col):
        """Return col, quickly."""
        try:
            value = self.attrs['metaarray'].data[col]
        except KeyError:
            value = self._get_frame(col)
            self.attrs['metaarray'].data[col] = value
        return value

    def _get_frame(self, col):
        """Return col from frame."""
        try:
            value = super().__getitem__(col).to_numpy()
        except (AttributeError, KeyError):
            value = np.full(len(self), self.metaframe.default[col])
            super().__setitem__(col, value)
        return self.format_value(col, value)


if __name__ == '__main__':

    dataarray = DataArray({'x': range(12)}, link=True,
                          Required=['x'], Array=['x'])
    dataarray.x = 7.7
    print(dataarray.loc[:, 'x'])
