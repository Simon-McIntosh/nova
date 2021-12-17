"""Manage subframe access."""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from nova.electromagnetic.framedata import FrameData
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

    def __getattr__(self, attr):
        """Return frame attribute."""
        return getattr(self.frame, attr)


@dataclass
class ArrayLocIndexer:
    """Access views of cached metaframe.data arrays."""

    name: str
    frame: FrameSpace = field(repr=False)
    subspace: bool = field(init=False, repr=False)
    attrs: list[str] = field(init=False, default_factory=list)
    _data: dict = field(init=False, repr=False)

    def __post_init__(self):
        """Set subspace flag and referance data arrays."""
        self.subspace = self.name[0] == 's'  # set subspace flag
        self.attrs = self.frame.attrs['metaframe'].array
        self.relink()

    def relink(self):
        """Relink data."""
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

    version: dict = field(init=False, default_factory=dict)
    ALoc: ArrayLocIndexer = field(init=False, repr=False)
    sALoc: ArrayLocIndexer = field(init=False, repr=False)
    aloc: ArrayLocIndexer = field(init=False, repr=False)
    saloc: ArrayLocIndexer = field(init=False, repr=False)
    Ic: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        """Create array loc indexers."""
        self.ALoc = ArrayLocIndexer('Array', self.frame)
        self.sALoc = ArrayLocIndexer('sArray', self.frame.subspace)
        self.aloc = ArrayLocIndexer('array', self.subframe)
        self.saloc = ArrayLocIndexer('sarray', self.subframe.subspace)
        self.Ic = self.saloc['Ic']
        self.version['frameloc'] = self.frame.version['index']
        self.version['subframeloc'] = self.subframe.version['index']

    def update_frameloc(self):
        """Update frame array loc indexer."""
        if self.version['frameloc'] != self.frame.version['index']:
            self.version['frameloc'] = self.frame.version['index']
            self.ALoc.relink()
            self.sALoc.relink()

    def update_subframeloc(self):
        """Update subframe array loc indexer."""
        if self.version['subframeloc'] != self.subframe.version['index']:
            self.version['subframeloc'] = self.subframe.version['index']
            self.aloc.relink()
            self.saloc.relink()
            self.Ic = self.saloc['Ic']

    def update_indexer(self):
        """Update links to array loc indexer following changes to index id."""
        self.update_frameloc()
        self.update_subframeloc()

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
