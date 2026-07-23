"""Manage subframe access."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import xxhash

from nova.frame.arraylocindexer import ArrayLocIndexer
from nova.frame.columnar import Vector, is_list_like
from nova.frame.framedata import FrameData
from nova.frame.framespace import FrameSpace
from nova.frame.error import SpaceKeyError, SubSpaceKeyError
from nova.frame.indexer import LabelAccessor


@dataclass
class RowSelection:
    """A labelled row selection exposing its index (Loc[rows, :])."""

    frame: FrameSpace
    rows: object

    @property
    def index(self):
        """Return the labels of the selected rows."""
        return Vector(np.asarray(self.frame.index)[self.rows])


@dataclass
class ColumnSelection:
    """A multi-column selection (Loc[rows, [cols]]) with squeeze support."""

    columns: dict

    def squeeze(self):
        """Return the single column when there is one, else self."""
        if len(self.columns) == 1:
            return next(iter(self.columns.values()))
        return self

    def to_list(self):
        """Return the selected columns as lists."""
        return [column.tolist() for column in self.columns.values()]


@dataclass
class LocIndexer:
    """Access frame / subframe attributes with a columnar loc indexer.

    ``Loc``/``loc`` target the full frame / subframe; ``sLoc``/``sloc`` (name
    starting with 's') target the subspace projection. Reads of a subspace
    column on a full-frame accessor inflate; direct writes are rejected in
    favour of the subspace accessor.
    """

    name: str
    frame: FrameSpace
    subspace: bool = field(init=False)

    def __post_init__(self):
        """Set the subspace-target flag from the accessor name."""
        self.subspace = self.name[0] == "s"

    def __call__(self):
        """Return the underlying frame."""
        return self.frame

    def __len__(self):
        """Return the frame length."""
        return len(self.frame)

    @staticmethod
    def _split(key):
        """Return (rows, column) from a loc key."""
        if isinstance(key, tuple):
            return key[0], key[1]
        return slice(None), key

    def _check(self, col, setting):
        """Enforce subspace membership / protection for a column access."""
        if not isinstance(col, str):
            return
        if self.subspace:
            if not self.frame.hascol("subspace", col):
                raise SubSpaceKeyError(col, self.frame.metaframe.subspace)
            return
        if setting and self.frame.hascol("subspace", col):
            raise SpaceKeyError(self.name, col)
        if setting:
            if col not in self.frame:
                self.frame.check_column(col)  # raises ColumnError for schema cols
                raise KeyError(col)
        elif col not in self.frame and not self.frame.hascol("subspace", col):
            raise KeyError(col)

    def __getitem__(self, key):
        """Return the selected column values (or a RowSelection for [rows, :])."""
        rows, col = self._split(key)
        if isinstance(col, slice):
            return RowSelection(self.frame, LabelAccessor(self.frame)._rows(rows))
        if is_list_like(col):
            return ColumnSelection({name: self.frame.loc[rows, name] for name in col})
        self._check(col, setting=False)
        return self.frame.loc[rows, col]

    def __setitem__(self, key, value):
        """Assign to the selected column values, in place on the store."""
        rows, col = self._split(key)
        self._check(col, setting=True)
        index = LabelAccessor(self.frame)._rows(rows)
        column = self.frame[col]
        if isinstance(index, slice) and index == slice(None, None, None):
            column[:] = value  # length-checked in place (ValueError on mismatch)
        else:
            column[index] = value


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
