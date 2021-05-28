"""Manage subframe access."""
from dataclasses import dataclass, field

import numpy.typing as npt

from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.error import SpaceKeyError


@dataclass
class LocIndexer:
    """Access frame attributes using a pandas style loc indexer."""

    name: str
    frame: FrameSpace
    subspace: bool = field(init=False)

    def __post_init__(self):
        """Set subspace flag."""
        self.subspace = self.name[0] == 's'

    def __call__(self):
        """Return frame attribute."""
        return self.frame

    def __len__(self):
        """Return frame length."""
        return len(self.frame)

    def __setitem__(self, key, value):
        """Set frame attribute."""
        index = self.frame.get_index(key)
        col = self.frame.get_col(key)
        if self.frame.hascol('subspace', col) and not self.subspace:
            raise SpaceKeyError(self.name, col)
        if self.frame.hascol('array', col):
            if isinstance(index, slice):
                if index == slice(None):
                    return setattr(self.frame, col, value)
            getattr(self.frame, col)[index] = value
            return None
        self.frame.loc[index, col] = value

    def __getitem__(self, key) -> npt.ArrayLike:
        """Return frame attribute."""
        index = self.frame.get_index(key)
        col = self.frame.get_col(key)
        if self.frame.hascol('array', col):
            if isinstance(index, slice):
                if index == slice(None):
                    return getattr(self.frame, col)
            self.frame.get_index(key)
            return getattr(self.frame, col)[index]
        return self.frame.loc[index, col]


@dataclass
class FrameSetLoc:
    """
    FrameSet Loc indexer.

        - Loc: Access frame attributes.
        - sLoc: Access frame subspace attributes.
        - loc: Access subframe attributes.
        - sloc: Access subframe subspace attributes.

    """

    frame: FrameSpace = field(default=None, repr=False)
    subframe: FrameSpace = field(default=None, repr=False)

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
