
import pandas
import numpy as np


class SubSpaceIndexError(IndexError):
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
            if self.obj.metaframe.lock:
                raise SubSpaceIndexError(self.name, col)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        col = self._get_col(key)
        if self.obj.is_subspace(col):
            if self.obj.metaframe.lock:
                self.obj.set_frame(col)
        return super().__getitem__(key)


class _ScalarAccessIndexer(IndexerMixin,
                           pandas.core.indexing._ScalarAccessIndexer):
    pass


class _LocationIndexer(IndexerMixin,
                       pandas.core.indexing._LocationIndexer):
    pass


class _iLocIndexer(_LocationIndexer,
                   pandas.core.indexing._iLocIndexer):
    pass


class _LocIndexer(_LocationIndexer,
                  pandas.core.indexing._LocIndexer):
    pass


class _AtIndexer(_ScalarAccessIndexer,
                 pandas.core.indexing._AtIndexer):
    pass


class _iAtIndexer(_ScalarAccessIndexer,
                  pandas.core.indexing._iAtIndexer):
    pass


class DataSpaceMixin:

    @property
    def loc(self) -> "_LocIndexer":
        """Extend DataFrame.loc, restrict subspace access."""
        return _LocIndexer("loc", self)

    @property
    def iloc(self) -> "_iLocIndexer":
        """Extend DataFrame.iloc, restrict subspace access."""
        return _iLocIndexer("iloc", self)

    @property
    def at(self) -> "_AtIndexer":
        """Extend DataFrame.at, restrict subspace access."""
        return _AtIndexer("at", self)

    @property
    def iat(self) -> "_AtIndexer":
        """Extend DataFrame.iat, restrict subspace access."""
        return _iAtIndexer("iat", self)

    def is_subspace(self, col):
        """Return Ture if col in subspace."""
        if isinstance(col, int):
            col = self.columns[col]
        if 'metaframe' in self.attrs and 'subspace' in self.attrs and \
                isinstance(col, str):
            return col in self.metaframe.subspace
        return False

    def check_subspace(self, col):
        """Check for col in metaframe.subspace, raise error if not found."""
        if not self.is_subspace(col):
            raise IndexError(f'\'{col}\' not specified in metaframe.subspace '
                             f'{self.metaframe.subspace}')

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.check_subspace(col)
        with self.metaframe.setlock(True):
            value = getattr(self, col).to_numpy()[self.subref]
        with self.metaframe.setlock(None):
            super().__setattr__(col, value)

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.check_subspace(col)
        with self.metaframe.setlock(False):
            return super().__getattr__(col)

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if hasattr(self, 'subspace'):
            for col in self.subspace:
                self.set_frame(col)

    def __getattr__(self, col):
        """Extend DataFrame.__getattr__. (frame.*)."""
        #if col in self.attrs:
        #    return self.attrs[col]
        if self.is_subspace(col):
            if self.metaframe.lock is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock is False:
                self.set_frame(col)
        return super().__getattr__(col)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.is_subspace(col):
            if self.metaframe.lock is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock is False:
                self.set_frame(col)
        return super().__getitem__(col)

    def __setattr__(self, col, value):
        """Check lock. Extend DataFrame.__setattr__ (frame.* = *).."""
        value = self._format_value(col, value)
        if col in self._attributes:
            self.attrs[col] = value
            return None
        if self.is_subspace(col):
            if self.metaframe.lock is True:
                return self.subspace.__setattr__(col, value)
            if self.metaframe.lock is False:
                raise SubSpaceIndexError('setattr', col)
        return super().__setattr__(col, value)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self._format_value(col, value)
        if self.is_subspace(col):
            if self.metaframe.lock is True:
                return self.subspace.__setitem__(col, value)
            if self.metaframe.lock is False:
                raise SubSpaceIndexError('setitem', col)
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
