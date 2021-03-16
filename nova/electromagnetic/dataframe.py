"""Subclass pandas.DataFrame."""
from typing import Optional, Collection, Any

import pandas
import numpy as np

from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.indexer import Indexer
from nova.electromagnetic.energize import Energize
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon

# pylint: disable=too-many-ancestors


class SubSpaceError(IndexError):
    """Prevent direct access to frame's subspace variables."""

    def __init__(self, name, col):
        super().__init__(
            f'{name} access is restricted for subspace attributes. '
            f'Use frame.subspace.{name}[:, {col}] = *.\n\n'
            'Lock may be overridden via the following context manager '
            'but subspace will still overwrite (Cavieat Usor):\n'
            'with frame.metaframe.setlock(None):\n'
            f'    frame.{name}[:, {col}] = *')


class IndexerMixin:
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj._get_col(key)
        value = self.obj._format_value(col, value)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                raise SubSpaceError(self.name, col)
        if self.obj.in_field(col, 'energize'):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        col = self.obj._get_col(key)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                self.obj.set_frame(col)
        if self.obj.in_field(col, 'energize'):
            if self.obj.metaframe.lock('subspace') is not None:
                return self.obj.energize._get_item(super(), key)
        return super().__getitem__(key)


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

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None):
        super().__init__(data, index, columns)
        self.attrs['indexer'] = Indexer(IndexerMixin)
        self.attrs['metaframe'] = MetaFrame()

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def loc(self):
        """Extend DataFrame.loc, restrict subspace access."""
        return self.indexer.loc("loc", self)

    @property
    def iloc(self):
        """Extend DataFrame.iloc, restrict subspace access."""
        return self.indexer.iloc("iloc", self)

    @property
    def at(self):
        """Extend DataFrame.at, restrict subspace access."""
        return self.indexer.at("at", self)

    @property
    def iat(self):
        """Extend DataFrame.iat, restrict subspace access."""
        return self.indexer.iat("iat", self)

    @staticmethod
    def isframe(obj, dataframe=True):
        """
        Return isinstance(arg[0], Frame | DataFrame) flag.

        Parameters
        ----------
        obj : Any
            Input.
        dataframe : bool, optional
            Accept pandas.DataFrame. The default is True.

        Returns
        -------
        isframe: bool
            Frame / pandas.DataFrame isinstance flag.

        """
        if isinstance(obj, DataFrame):
            return True
        if isinstance(obj, pandas.DataFrame) and dataframe:
            return True
        return False

    def _get_col(self, key):
        """Return column label."""
        if isinstance(key, tuple):
            col = key[-1]
        else:
            col = key
        if isinstance(col, int):
            col = self.columns[col]
        return col

    def _hasattr(self, attr):
        """Return True if attr in self.attrs."""
        return attr in self.attrs

    def in_field(self, col, field):
        """Return Ture if col in metaframe.{field} and hasattr(self, field)."""
        if not isinstance(col, str):
            return False
        if self._hasattr('metaframe') and self._hasattr(field):
            if hasattr(self.attrs[field], 'columns'):
                return col in self.attrs[field].columns
        #if self._hasattr('metaframe') and field == 'subspace':
        return False

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if self._hasattr('subspace'):
            for col in self.subspace:
                self.set_frame(col)

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(True, 'subspace'):
            value = getattr(self, col).to_numpy()[self.subref]
        with self.metaframe.setlock(None, 'subspace'):
            super().__setitem__(col, value)

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(False, 'subspace'):
            return super().__getitem__(col)

    def assert_in_field(self, col, field):
        """Check for col in metaframe.{field}, raise error if not found."""
        try:
            self.in_field(col, field)
        except AssertionError as in_field_assert:
            raise AssertionError(
                f'\'{col}\' not specified in metaframe.subspace '
                f'{self.metaframe.subspace}') from in_field_assert

    def __getattr__(self, name):
        """Extend DataFrame.__getattr__. (frame.*)."""
        if name in self.attrs:
            return self.attrs[name]
        return super().__getattr__(name)

    def __getitem__(self, key):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        col = self._get_col(key)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('subspace') is not None:
                return self.energize._get_item(super(), key)
        return super().__getitem__(col)

    def __setitem__(self, key, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        col = self._get_col(key)
        value = self._format_value(col, value)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setitem__(key, value)
            if self.metaframe.lock('subspace') is False:
                raise SubSpaceError('setitem', col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is False:
                return self.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def _format_value(self, col, value):
        if not pandas.api.types.is_numeric_dtype(type(value)) \
                or not self._hasattr('metaframe'):
            return value
        try:
            dtype = type(self.metaframe.default[col])
        except KeyError:  # no default type
            return value
        try:
            if pandas.api.types.is_list_like(value):
                return np.array(value, dtype)
            return dtype(value)
        except (ValueError, TypeError):  # NaN conversion error
            return value


if __name__ == '__main__':

    df = DataFrame({'x': [1, 2, 3], 'z': 6.7})
    df.loc[:, 'x'] = 7
    print(df.loc[:, 'x'])
