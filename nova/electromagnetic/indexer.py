"""Extend pandas Indexer methods."""
import pandas

# pylint: disable=protected-access
# pylint: disable=invalid-name


class Indexer:
    """Extend pandas Indexer methods."""

    def __init__(self, *mixins):
        """Set mixin to extend get/setitem methods (scalaraccess, location)."""
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


class LocMinin:
    """Extend pandas.DataFrame indexer methods."""

    def loc_mixin(self, *mixins):
        """Init indexer method, extend getitem and setitem using mixins."""
        if not self.hasattrs('indexer'):
            self.attrs['indexer'] = Indexer(*mixins)
        else:
            mixins.extend(self.attrs['indexer'].mixin)
            self.attrs['indexer'] = Indexer(*mixins)

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


