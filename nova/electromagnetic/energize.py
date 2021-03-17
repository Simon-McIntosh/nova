"""Manage frame currents."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nova.electromagnetic.metamethod import MetaMethod

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame


@dataclass
class Energize(MetaMethod):
    """Manage dependant frame energization parameters."""

    frame: Frame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['Ic', 'It', 'Nt'])
    additional: list[str] = field(default_factory=lambda: [
        'Ic', 'It', 'Nt', 'active', 'plasma', 'optimize', 'feedback'])
    require_all: bool = False

    def initialize(self):
        """Init attributes."""
        pass

    @property
    def columns(self):
        """Return required columns."""
        return self.required

    def _get_key(self, key, col=None):
        if col is None:
            return key
        if isinstance(key, tuple):
            if isinstance(key[-1], int):
                if not isinstance(col, int):
                    col = self.frame.columns.get_loc(col)
            return (*key[:-1], col)
        return col

    def _hasattrs(self, attrs: list[str]):
        """Return True if all required attributes are available else False."""
        return np.array([attr in self.frame.columns for attr in attrs]).all()

    def _set_item(self, indexer, key, value):
        if self.frame._get_col(key) == 'It' and \
                self.frame.metaframe.lock('energize') is not True:
            value /= indexer.__getitem__(self._get_key(key, 'Nt'))
            return indexer.__setitem__(self._get_key(key, 'Ic'), value)
        return indexer.__setitem__(key, value)

    def _get_item(self, indexer, key):
        if self.frame._get_col(key) == 'It' and self._hasattrs(['Ic', 'Nt']):
            line_current = indexer.__getitem__(self._get_key(key, 'Ic'))
            turn_number = indexer.__getitem__(self._get_key(key, 'Nt'))
            with self.frame.metaframe.setlock(True, 'energize'):
                self._set_item(indexer, key, line_current*turn_number)
        return indexer.__getitem__(key)

    '''
    def __setitem__(self, key, value):
        """Manage setattr for dependant variables."""
        if hasattr(self, 'frame'):
            col = self.frame._get_col(key)
            if self.frame.in_field(col, 'energize'):
                return self._update(key, value, col)
        return super().__setitem__(key, value)

    def is_integer_slice(self, index):
        """Return True if slice.start or slice.stop is int."""
        if not isinstance(index, slice):
            return False
        return isinstance(index.start, int) or isinstance(index.stop, int)

    def _set_item(self, key, value, col=None):
        key = self._get_key(key, col)
        if isinstance(key, str):
            self.frame[key] = value
        elif self.is_integer_slice(key[0]):
            self.frame.iloc[key] = value
        else:
            self.frame.loc[key] = value

    def _get_item(self, key, col=None):
        key = self._get_key(key, col)
        if isinstance(key, str):
            return self.frame[key]
        if self.is_integer_slice(key[0]):
            return self.frame.iloc[key]
        print(key)
        return self.frame.loc[key]

    def _update(self, key, value, col):
        """Set col, update dependant parameters."""
        with self.frame.metaframe.setlock(True, 'energize'):
            self._set_item(key, value)
            if col == 'Ic':  # line current  -> update turn current
                value *= self._get_item(key, 'Nt').values
                self._set_item(key, value, 'It')
            if col == 'It':  # turn current  -> update line current
                value /= self._get_item(key, 'Nt').values
                self._set_item(key, value, 'Ic')
            if col == 'Nt':  # turn number  -> update turn current
                value *= self._get_item(key, 'Ic').values
                self._set_item(key, value, 'It')
    '''
