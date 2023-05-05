"""Manage fast access dataframe attributes."""
import numpy as np
import pandas

from nova.frame.dataframe import DataFrame
from nova.frame.indexer import Indexer

# pylint: disable=too-many-ancestors


class ArrayLocMixin():
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Extend Loc setitem."""
        col = self.obj.get_col(key)
        key = self.obj.get_key(key)
        if self.obj.hascol('array', col):
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
        key = self.obj.get_key(key)
        if self.obj.hascol('array', col):
            if self.obj.lock('array') is False:
                try:
                    return super().__getitem__(key)
                except AttributeError:  # loc[slice, col]
                    index = self.obj.get_index(key)
                    return self.obj.__getitem__(col)[index]
        return super().__getitem__(key)


class ArrayIndexer(Indexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return ArrayLocMixin


class DataArray(ArrayIndexer, DataFrame):
    """Extend DataFrame enabling fast access to dynamic fields in array."""

    def __init__(self, data=None, index=None, columns=None,
                 attrs=None, **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.unlink_array()

    def __repr__(self):
        """Propagate array variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def unlink_array(self):
        """Unlink items in fast access dataarray cache."""
        attrs = list(self.metaframe.data)
        unlink = {attr: self.metaframe.data.pop(attr) for attr in attrs
                  if self.metaframe.data[attr].shape[0] != self.shape[0]}
        self.overwrite_array(unlink)
        return self

    def overwrite_array(self, data=None):
        """Overwrite float items in data with nans."""
        if data is None:
            data = self.metaframe.data
        for attr in data:
            data[attr][:] = np.nan

    def update_frame(self):
        """Extend DataFrame.update_frame, transfer metaarray.data to frame."""
        for col in self.metaframe.array:
            if col in self:
                self._set_frame(col)

    def __setattr__(self, col, value):
        """Extend DataFrame.__setattr__ to gain fast access to array data."""
        if self.hascol('array', col):
            return self._set_array(col, value)
        return super().__setattr__(col, value)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.hascol('array', col):
            if self.lock('array') is False:
                return self._get_array(col)
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Extend DataFrame.__setitem__. (frame['*'] = *)."""
        if self.hascol('array', col):
            if self.lock('array') is False:
                return self._set_array(col, value)
        return super().__setitem__(col, value)

    def _get_array(self, col):
        """Return col, quickly."""
        try:
            return self.attrs['metaframe'].data[col]
        except KeyError:
            value = self.format_value(col, self._get_frame(col))
            self.attrs['metaframe'].data[col] = value
            return value

    def _set_array(self, col, value):
        """Set col, quickly, preserving shape."""
        try:
            self.attrs['metaframe'].data[col][:] = value
        except KeyError as keyerror:
            if not pandas.api.types.is_list_like(value):
                value = self.format_value(col, value)
                value = np.full(len(self), value, dtype=type(value))
            elif len(value) != len(self):
                raise IndexError(f'input length {len(value)} != {len(self)}') \
                    from keyerror
            self.attrs['metaframe'].data[col] = np.zeros_like(value)
            self._set_array(col, value)

    def _get_frame(self, col):
        """Return col from frame."""
        try:
            value = np.array(super().__getitem__(col).to_numpy())
        except (AttributeError, KeyError):
            value = np.full(len(self), self.metaframe.default[col])
            super().__setitem__(col, value)
        return value

    def _set_frame(self, col):
        """Transfer metaarray.data to frame."""
        self.metaframe.assert_hascol('array', col)
        value = self._get_array(col)
        with self.setlock(True, 'array'):
            super().__setitem__(col, value)

    def store(self, file, group=None, mode='w', vtk=False):
        """Store dataframe as group in netCDF4 hdf5 file."""
        self.update_frame()
        super().store(file, group, mode, vtk=vtk)


if __name__ == '__main__':

    dataarray = DataArray({'x': range(7), 'z': 0},
                          additional=['Ic'], Array=['Ic'], label='Coil')

    dataarray = DataArray({'x': range(7)},
                          additional=['Ic', 'free'], array=['x'], label='Coil')

    print(dataarray.free)
