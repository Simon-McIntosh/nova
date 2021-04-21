"""Manage subframe access."""
from dataclasses import dataclass, field

import numpy.typing as npt

from nova.electromagnetic.dataarray import DataArray
from nova.electromagnetic.frame import Frame


@dataclass
class LocIndexer:
    """Access frame attributes."""

    name: str
    frame: DataArray

    def unpack(self, key):
        """Unpack item access key."""
        if isinstance(key, tuple):
            return self._getindex(key[0]), key[1]
        return slice(None), key

    def _getindex(self, index):
        if isinstance(index, str):
            if index in self.frame:
                return getattr(self.frame, index)
        return index

    def __setitem__(self, key, value):
        """Set frame attribute."""
        index, col = self.unpack(key)
        if self.frame.metaframe.hascol('array', col):
            if isinstance(index, slice):
                if index == slice(None):
                    return setattr(self.frame, col, value)
            getattr(self.frame, col)[index] = value
            return None
        self.frame.loc[index, col] = value

    def __getitem__(self, key) -> npt.ArrayLike:
        """Return frame attribute."""
        index, col = self.unpack(key)
        if self.frame.metaframe.hascol('array', col):
            if isinstance(index, slice):
                if index == slice(None):
                    return getattr(self.frame, col)
            return getattr(self.frame, col)[index]
        return self.frame.loc[index, col]


@dataclass
class FrameLoc:
    """
    FrameSet Loc indexer.

        - Loc: Access frame attributes.
        - sLoc: Access frame subspace attributes.
        - loc: Access subframe attributes.
        - sloc: Access subframe subspace attributes.

    """

    frame: Frame = field(init=False, default=None, repr=False)
    subframe: Frame = field(init=False, default=None, repr=False)

    @property
    def Loc(self):
        """Access frame attributes."""
        return LocIndexer('Loc', self.frame)

    @property
    def sLoc(self):
        """Access subspace frame attributes."""
        return LocIndexer('sLoc', self.frame.subspace)

    @property
    def loc(self):
        """Access subframe attributes."""
        return LocIndexer('loc', self.subframe)

    @property
    def sloc(self):
        """Access subspace subframe attributes."""
        return LocIndexer('sloc', self.subframe.subspace)
