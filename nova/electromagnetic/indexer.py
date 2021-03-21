"""Extend pandas Indexer methods."""
from abc import ABC, abstractmethod

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
            '_ScalarAccessIndexer',
            (*self.mixin, pandas.core.indexing._ScalarAccessIndexer), {})

    def location(self):
        """Return _LocationIndexer."""
        return type(
            '_LocationIndexer',
            (*self.mixin, pandas.core.indexing._LocationIndexer), {})

    def iloc(self, *args):
        """Return _iLocIndexer."""
        return type(
            '_iLocIndexer',
            (self.location(), pandas.core.indexing._iLocIndexer), {})(*args)

    def loc(self, *args):
        """Return _LocIndexer."""
        return type(
            '_LocIndexer',
            (self.location(), pandas.core.indexing._LocIndexer), {})(*args)

    def at(self, *args):
        """Return _AtIndexer."""
        return type(
            '_AtIndexer',
            (self.scalaraccess(), pandas.core.indexing._AtIndexer), {})(*args)

    def iat(self, *args):
        """Return _iAtIndexer."""
        return type(
            '_iAtIndexer',
            (self.scalaraccess(), pandas.core.indexing._iAtIndexer), {})(*args)


class Indexer(ABC):
    """Extend pandas.DataFrame indexer methods."""

    def extract_attrs(self, data, attrs):
        """Extend DataFrame.extract_attrs, insert metaarray."""
        super().extract_attrs(data, attrs)
        if not self.hasattrs('indexer'):
            self.attrs['indexer'] = LocIndexer(self.loc_mixin)  # init indexer

    @property
    @abstractmethod
    def loc_mixin(self) -> object:
        """Return LocIndexer mixin."""

    @property
    def loc(self):
        """Extend DataFrame.loc, restrict subspace access."""
        return self.indexer.loc("loc", self)

    @property
    def iloc(self):
        """Extend DataFrame.iloc, restrict subspace access."""
        return self.indexer.iloc("iloc", self)

    @property
    def at(self):
        """Extend DataFrame.at, restrict subspace access."""
        return self.indexer.at("at", self)

    @property
    def iat(self):
        """Extend DataFrame.iat, restrict subspace access."""
        return self.indexer.iat("iat", self)
