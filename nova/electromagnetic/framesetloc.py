"""Manage subframe access."""
from dataclasses import dataclass, field

import numpy.typing as npt

from nova.electromagnetic.framedata import FrameData
from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.error import SpaceKeyError


@dataclass
class LocIndexer:
    """Access frame attributes using a pandas style loc indexer."""

    name: str
    frame: FrameSpace
    subspace: bool = False

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
class ArrayLocIndexer:
    """Access views of cached metaframe.data arrays."""

    name: str
    frame: FrameSpace = field(repr=False)
    subspace: bool = False
    version: int = field(init=False)
    attrs: list[str] = field(init=False)
    _data: dict = field(init=False, repr=False)

    def __post_init__(self):
        """Extract frame.index version and referance data arrays."""
        self.version = id(self.frame.index)
        self.attrs = self.frame.attrs['metaframe'].array
        self._data = {attr: self.frame[attr] for attr in self.attrs}

    def __setitem__(self, key, value):
        """Set data array item in metaframe.data dict."""
        self._data[key][:] = value

    def __getitem__(self, key) -> npt.ArrayLike:
        """Return data array item in metaframe.data dict."""
        return self._data[key]

    def __getattr__(self, attr):
        """Implement fast attribute lookup."""
        return self._data[attr]


@dataclass
class FrameSetLoc(FrameData):
    """
    FrameSet Loc indexer.

        - Loc: Access frame attributes.
        - sLoc: Access frame subspace attributes.
        - loc: Access subframe attributes.
        - sloc: Access subframe subspace attributes.

    """

    def __post_init__(self):
        """Create Loc indexers."""
        #super().__post_init__()

    @property
    def Loc(self):
        """Access frame attributes."""
        return LocIndexer('Loc', self.frame)

    @property
    def sLoc(self):
        """Access subspace frame attributes."""
        return LocIndexer('sLoc', self.frame.subspace, True)

    @property
    def loc(self):
        """Access subframe attributes."""
        return LocIndexer('loc', self.subframe)

    @property
    def sloc(self):
        """Access subspace subframe attributes."""
        return LocIndexer('sloc', self.subframe.subspace, True)

    @property
    def ALoc(self) -> dict:
        """Return view of frame array attributes."""
        return ArrayLocIndexer('Array', self.frame)

    @property
    def sALoc(self) -> dict:
        """Access frame subspace array attributes."""
        return ArrayLocIndexer('sArray', self.frame.subspace, True)

    @property
    def aloc(self) -> dict:
        """Access subframe array attributes."""
        return ArrayLocIndexer('array', self.subframe)

    @property
    def saloc(self) -> dict:
        """Access subframe subspace array attributes."""
        return ArrayLocIndexer('sarray', self.subframe.subspace, True)
