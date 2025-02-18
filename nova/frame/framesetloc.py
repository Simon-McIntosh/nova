"""Manage subframe access."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import xxhash

from nova.frame.arraylocindexer import ArrayLocIndexer
from nova.frame.framedata import FrameData
from nova.frame.framespace import FrameSpace
from nova.frame.error import SpaceKeyError


@dataclass
class LocIndexer:
    """Access frame attributes using a pandas style loc indexer."""

    name: str
    frame: FrameSpace
    subspace: bool = field(init=False)

    def __post_init__(self):
        """Set subspace flag."""
        self.subspace = self.name[0] == "s"

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
        if self.frame.hascol("subspace", col) and not self.subspace:
            raise SpaceKeyError(self.name, col)
        if self.frame.hascol("array", col):
            if isinstance(index, slice) and index == slice(None):
                return setattr(self.frame, col, value)
            getattr(self.frame, col)[index] = value
            return None
        self.frame.loc[index, col] = value

    def __getitem__(self, key) -> np.ndarray:
        """Return frame attribute."""
        index = self.frame.get_index(key)
        col = self.frame.get_col(key)
        if self.frame.hascol("array", col):
            if isinstance(index, slice) and index == slice(None):
                return getattr(self.frame, col)
            self.frame.get_index(key)
            return getattr(self.frame, col)[index]
        self.frame.update_frame()
        if isinstance(index, int):
            index = self.frame.index[index]
        return self.frame.loc[index, col]


@dataclass
class HashLoc:
    """Data Loc base class."""

    name: str
    aloc: ArrayLocIndexer = field(repr=False)
    saloc: ArrayLocIndexer | None = field(repr=False, default=None)
    xxh64: xxhash.xxh64 = field(repr=False, init=False)
    subspace: list[str] = field(repr=False, init=False)

    def __post_init__(self):
        """Create xxhash generator."""
        self.xxh64 = xxhash.xxh64()
        try:
            self.subspace = self.saloc.frame.columns.to_list()
        except AttributeError:
            self.subspace = []

    def _array(self, key):
        """Return loc array."""
        if key in self.subspace:
            return self.saloc[key]
        return self.aloc[key]

    def __getitem__(self, key) -> int:
        """Return interger has computed on aloc data array item."""
        self.xxh64.reset()
        self.xxh64.update(self._array(key))
        return self.xxh64.intdigest()


@dataclass
class FrameSetLoc(FrameData):
    """
    FrameSet Loc indexer.

        - Loc: Access frame attributes.
        - sLoc: Access frame subspace attributes.
        - loc: Access subframe attributes.
        - sloc: Access subframe subspace attributes.

    """

    version: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        """Create array loc indexers."""
        self.version |= dict(frameloc=None, subframeloc=None)
        self.update_loc_indexer()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def _clear_cache(self, attrs: list[str]):
        """Clear cached properties."""
        for attr in attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def update_aloc_hash(self, attr):
        """Update subframe version."""
        self.subframe.version[attr] = self.aloc_hash[attr]

    def update_frameloc(self):
        """Update frame array loc indexer."""
        if self.version["frameloc"] != self.frame.version["index"]:
            self.version["frameloc"] = self.frame.version["index"]
            self._clear_cache(["ALoc", "sALoc"])

    def update_subframeloc(self):
        """Update subframe array loc indexer."""
        if self.version["subframeloc"] != self.subframe.version["index"]:
            self.version["subframeloc"] = self.subframe.version["index"]
            self._clear_cache(["aloc", "saloc", "aloc_hash"])

    def update_loc_indexer(self):
        """Update links to array loc indexer following changes to index id."""
        self.update_frameloc()
        self.update_subframeloc()

    @cached_property
    def plasma_index(self):
        """Return plasma index."""
        try:
            return next(
                self.frame.subspace.index.get_loc(name)
                for name in self.subframe.frame[self.aloc["plasma"]].unique()
            )
        except StopIteration:
            return -1

    @property
    def i_plasma(self):
        """Return total plasma current."""
        return self.saloc["Ic"][self.plasma_index]

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.i_plasma)

    @cached_property
    def coil_name(self):
        """Return coil names."""
        return np.array([name for name in self.Loc["coil", :].index])

    @cached_property
    def _subref(self):
        """Return frame current subframe reference."""
        return self.Loc["coil", "subref"].to_numpy()

    @cached_property
    def _factor(self):
        """Return frame current link factor."""
        return self.Loc["coil", "factor"].to_numpy()

    @property
    def current(self):
        """Return frame coil currents."""
        return self._factor * self.saloc["Ic"][self._subref]

    @cached_property
    def aloc_hash(self):
        """Return interger hash computed on aloc array attribute."""
        return HashLoc("array_hash", self.aloc, self.saloc)

    @cached_property
    def ALoc(self):
        """Return fast frame array attributes."""
        return ArrayLocIndexer("Array", self.frame)

    @cached_property
    def sALoc(self):
        """Return fast frame subspace array attributes."""
        return ArrayLocIndexer("sArray", self.frame.subspace)

    @cached_property
    def aloc(self):
        """Return fast subframe array attributes."""
        return ArrayLocIndexer("array", self.subframe)

    @cached_property
    def saloc(self):
        """Return fast subframe subspace array attributes."""
        return ArrayLocIndexer("sarray", self.subframe.subspace)

    @property
    def Loc(self):
        """Access frame attributes."""
        return LocIndexer("Loc", self.frame)

    @property
    def sLoc(self):
        """Access subspace frame attributes."""
        return LocIndexer("sLoc", self.frame.subspace)

    @property
    def loc(self):
        """Access subframe attributes."""
        return LocIndexer("loc", self.subframe)

    @property
    def sloc(self):
        """Access subspace subframe attributes."""
        return LocIndexer("sloc", self.subframe.subspace)
