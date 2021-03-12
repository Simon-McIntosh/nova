"""Subclass pandas.DataFrame."""
from typing import Optional, Collection, Any

import pandas
import numpy as np

from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.indexer import IndexerMixin, Indexer

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

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 mixin: IndexerMixin = IndexerMixin):
        super().__init__(data, index, columns)
        self.indexer = Indexer(mixin)
        self.attrs['metaframe'] = MetaFrame()

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
        col = self._get_col(col)
        if not isinstance(col, str):
            return False
        if self._hasattr('metaframe') and self._hasattr(field):
            return col in getattr(self.metaframe, field)
        return False

    def __getattr__(self, col):
        """Extend pandas.DataFrame.__getattr__. Intercept attrs."""
        if col in self.attrs:
            return self.attrs[col]
        return super().__getattr__(col)

    def __setattr__(self, col, value):
        """Extend pandas.Extend DataFrame.__setattr__ (frame.* = *).."""
        value = self._format_value(col, value)
        return super().__setattr__(col, value)

    def __setitem__(self, col, value):
        """Extend pandas.Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self._format_value(col, value)
        return super().__setitem__(col, value)

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
    df.x = 7
    print(df.loc[:, 'x'])
