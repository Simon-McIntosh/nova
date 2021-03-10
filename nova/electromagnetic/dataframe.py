"""Subclass pandas.DataFrame."""
from typing import Optional, Collection, Any

import pandas
import numpy as np


# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object
# pylint: disable=protected-access


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

    _attributes = ['multipoint', 'subspace', 'polygon']

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None):
        super().__init__(data, index, columns)

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    def __getattr__(self, col):
        """Extend pandas.DataFrame.__getattr__. Intercept attrs."""
        if col in self.attrs:
            return self.attrs[col]
