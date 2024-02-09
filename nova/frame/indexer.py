"""Extend pandas Indexer methods."""

from abc import ABC, abstractmethod
from functools import cached_property

import pandas

# pylint: disable=protected-access
# pylint: disable=invalid-name


class LocIndexer:
    """Extend pandas Indexer methods."""

    def __init__(self, *mixins):
        self.mixin = [mixin for mixin in mixins if mixin is not None]

    def scalaraccess(self):
        """Return _ScalarAccessIndexer."""
        return type(
            "_ScalarAccessIndexer",
            (*self.mixin, pandas.core.indexing._ScalarAccessIndexer),
            {},
        )

    def location(self):
        """Return _LocationIndexer."""
        return type(
            "_LocationIndexer", (*self.mixin, pandas.core.indexing._LocationIndexer), {}
        )

    def iloc(self, *args):
        """Return _iLocIndexer."""
        return type(
            "_iLocIndexer", (self.location(), pandas.core.indexing._iLocIndexer), {}
        )(*args)

    def loc(self, *args):
        """Return _LocIndexer."""
        return type(
            "_LocIndexer", (self.location(), pandas.core.indexing._LocIndexer), {}
        )(*args)

    def at(self, *args):
        """Return _AtIndexer."""
        return type(
            "_AtIndexer", (self.scalaraccess(), pandas.core.indexing._AtIndexer), {}
        )(*args)

    def iat(self, *args):
        """Return _iAtIndexer."""
        return type(
            "_iAtIndexer", (self.scalaraccess(), pandas.core.indexing._iAtIndexer), {}
        )(*args)


class Indexer(ABC):
    """Extend pandas.DataFrame indexer methods."""

    def extract_attrs(self, data, attrs):
        """Extend DataFrame.extract_attrs, insert metaarray."""
        super().extract_attrs(data, attrs)
        if not self.hasattrs("indexer"):
            self.attrs["indexer"] = LocIndexer(self.loc_mixin)  # init indexer

    @property
    @abstractmethod
    def loc_mixin(self) -> object:
        """Return LocIndexer mixin."""

    @cached_property
    def loc(self):
        """Extend DataFrame.loc, restrict subspace access."""
        return self.indexer.loc("loc", self)

    @cached_property
    def iloc(self):
        """Extend DataFrame.iloc, restrict subspace access."""
        return self.indexer.iloc("iloc", self)

    @cached_property
    def at(self):
        """Extend DataFrame.at, restrict subspace access."""
        return self.indexer.at("at", self)

    @cached_property
    def iat(self):
        """Extend DataFrame.iat, restrict subspace access."""
        return self.indexer.iat("iat", self)

    def get_col(self, key) -> str:
        """Return column label."""
        if isinstance(key, tuple):
            col = key[-1]
        else:
            col = key
        if isinstance(col, int):
            col = self.columns[col]
        return col

    def get_key(self, key):
        """Return formated key - implement str index formating."""
        if not isinstance(key, tuple):
            return key
        if not isinstance(key[0], str):
            return key
        index = self.get_index(key)
        col = self.get_col(key)
        if isinstance(index, int):
            return key[0], col
        return index, col

    def get_index(self, key) -> slice | list[int] | pandas.Index:
        """Return index."""
        if not isinstance(key, tuple):
            return slice(None)
        index = key[0]
        if not isinstance(index, slice):
            if isinstance(index, str):
                if index in self:
                    return getattr(self, index)
                try:
                    return self.index.get_loc(index)
                except KeyError as keyerror:
                    if "part" in self:
                        return index == self.part
                    raise KeyError from keyerror
            return index
        _slice = [0 for __ in range(3)]
        for i, location in enumerate(["start", "stop", "step"]):
            value = getattr(index, location)
            if isinstance(value, str):
                value = self.index.get_loc(value)
                if location == "stop":  # maintain consistency with pandas
                    value += 1
            _slice[i] = value
        return slice(*_slice)

    def get_subkey(self, key):
        """Return subspace column intersection."""
        columns = key[1]
        if isinstance(columns, slice):
            if columns == slice(None):
                return key
        try:
            columns = self.columns[columns]  # perform slice
            interger = True
        except IndexError:
            interger = False
        try:
            subcols = self.subspace.columns.intersection(columns).values
            if interger:
                subcols = [self.subspace.columns.get_loc(i) for i in subcols]
            return (key[0], subcols)
        except TypeError as type_error:  # single value
            subcol = columns
            if subcol not in self.subspace.columns:
                raise IndexError(
                    f"column index {subcol} not found in "
                    f"subspace {self.subspace.columns}"
                ) from type_error
            if interger:
                subcol = self.subspace.columns.get_loc(subcol)
            return (key[0], subcol)
