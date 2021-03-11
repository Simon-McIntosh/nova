"""Manage frame super/sub space. Protect access to subspace attrs via frame."""

from typing import Optional, Collection, Any, Union

import pandas
import numpy as np

from nova.electromagnetic.superframe import SuperFrame
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame

# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object
# pylint: disable=protected-access


class SuperSpaceIndexError(IndexError):
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
    """Protect subspace items from direct access using loc, iloc, at, iat."""

    def _get_col(self, key):
        """Return column label."""
        if isinstance(key, tuple):
            col = key[-1]
        else:
            col = key
        if isinstance(col, int):
            col = self.obj.columns[col]
        return col

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self._get_col(key)
        if self.obj.is_subspace(col):
            if self.obj.metaframe.lock('subspace'):
                raise SuperSpaceIndexError(self.name, col)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        col = self._get_col(key)
        if self.obj.is_subspace(col):
            if self.obj.metaframe.lock('subspace'):
                self.obj.set_frame(col)
        return super().__getitem__(key)


class Indexer:
    """Extend pandas Indexer methods via Indexer class factory."""

    @staticmethod
    def scalaraccess():
        """Return _ScalarAccessIndexer."""
        return type(
            '_ScalarAccessIndexer',
            (IndexerMixin, pandas.core.indexing._ScalarAccessIndexer), {})

    @staticmethod
    def location():
        """Return _LocationIndexer."""
        return type(
            '_LocationIndexer',
            (IndexerMixin, pandas.core.indexing._LocationIndexer), {})

    @classmethod
    def iloc(cls, *args):
        """Return _iLocIndexer."""
        return type(
            '_iLocIndexer',
            (cls.location(), pandas.core.indexing._iLocIndexer), {})(*args)

    @classmethod
    def loc(cls, *args):
        """Return _LocIndexer."""
        return type(
            '_LocIndexer',
            (cls.location(), pandas.core.indexing._LocIndexer), {})(*args)

    @classmethod
    def at(cls, *args):
        """Return _AtIndexer."""
        return type(
            '_AtIndexer',
            (cls.scalaraccess(), pandas.core.indexing._AtIndexer), {})(*args)

    @classmethod
    def iat(cls, *args):
        """Return _iAtIndexer."""
        return type(
            '_iAtIndexer',
            (cls.scalaraccess(), pandas.core.indexing._iAtIndexer), {})(*args)


class SuperSpace(SuperFrame):
    """
    Extend SuperFrame to implement super/sub space access.

    - Protect setitem operations on loc, iloc, at, and iat operators.
    - Extend set/getattr and set/getitem to serve subspace.
    - Extend repr to update superframe prior to printing.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Union[MetaArray, MetaFrame]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.attrs['subspace'] = SubSpace(self)

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    @property
    def loc(self):
        """Extend DataFrame.loc, restrict subspace access."""
        return Indexer.loc("loc", self)

    @property
    def iloc(self):
        """Extend DataFrame.iloc, restrict subspace access."""
        return Indexer.iloc("iloc", self)

    @property
    def at(self):
        """Extend DataFrame.at, restrict subspace access."""
        return Indexer.at("at", self)

    @property
    def iat(self):
        """Extend DataFrame.iat, restrict subspace access."""
        return Indexer.iat("iat", self)

    def is_subspace(self, col):
        """Return Ture if col in subspace."""
        if isinstance(col, int):
            col = self.columns[col]
        if not isinstance(col, str):
            return False
        if 'metaframe' in self.attrs and 'subspace' in self.attrs:
            return col in self.metaframe.subspace
        return False

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if hasattr(self, 'subspace'):
            for col in self.subspace:
                self.set_frame(col)

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.check_subspace(col)
        with self.metaframe.setlock('subspace', True):
            value = getattr(self, col).to_numpy()[self.subref]
        with self.metaframe.setlock('subspace', None):
            super().__setattr__(col, value)

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.check_subspace(col)
        with self.metaframe.setlock('subspace', False):
            return super().__getattr__(col)

    def check_subspace(self, col):
        """Check for col in metaframe.subspace, raise error if not found."""
        if not self.is_subspace(col):
            raise IndexError(f'\'{col}\' not specified in metaframe.subspace '
                             f'{self.metaframe.subspace}')

    def __getattr__(self, col):
        """Extend DataFrame.__getattr__. (frame.*)."""
        if col in self.attrs:
            return self.attrs[col]
        if self.is_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        return super().__getattr__(col)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.is_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        return super().__getitem__(col)

    def __setattr__(self, col, value):
        """Check lock. Extend DataFrame.__setattr__ (frame.* = *).."""
        value = self._format_value(col, value)
        if self.is_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setattr__(col, value)
            if self.metaframe.lock('subspace') is False:
                raise SuperSpaceIndexError('setattr', col)
        return super().__setattr__(col, value)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self._format_value(col, value)
        if self.is_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setitem__(col, value)
            if self.metaframe.lock('subspace') is False:
                raise SuperSpaceIndexError('setitem', col)
        return super().__setitem__(col, value)

    def _format_value(self, col, value):
        if not pandas.api.types.is_numeric_dtype(type(value)) \
                or 'metaframe' not in self.attrs:
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

    superspace = SuperSpace(Required=['x', 'z'], Additional=['Ic'])

    superspace.add_frame(4, range(3), link=True)
    '''
    superspace.add_frame(4, range(2), link=False)
    superspace.add_frame(4, range(40), link=True)

    superspace.subspace.loc[:, 'Ic'] = 12
    superspace.subspace.loc['Coil5', 'Ic'] = 5.4

    print(superspace)
    '''
