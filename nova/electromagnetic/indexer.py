"""Extend pandas Indexer methods."""
import pandas

# pylint: disable=protected-access
# pylint: disable=invalid-name


class Indexer:
    """Extend pandas Indexer methods."""

    def __init__(self, *mixin):
        """Set mixin to extend get/setitem methods (scalaraccess, location)."""
        self.mixin = [mix for mix in mixin if mix is not None]

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


if __name__ == '__main__':

    df = pandas.DataFrame()
