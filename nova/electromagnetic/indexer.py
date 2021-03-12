
import pandas


class SuperSpaceIndexError(IndexError):
    """Prevent direct access to frame's subspace variables."""

    def __init__(self, name, col):
        super().__init__(
            f'{name} access is restricted for subspace attributes. '
            f'Use frame.subspace.{name}[:, {col}] = *.\n\n'
            'Lock may be overridden via the following context manager '
            'but subspace will still overwrite (Cavieat Usor):\n'
            'with frame.metaframe.setlock(None):\n'
            f'    frame.{name}[:, {col}] = *')


class IndexerMixin:

    def __setitem__(self, key, value):
        """Extend indexer setitem."""
        print('set base loc', key)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Extend indexer getitem."""
        print('get base loc', key)
        return super().__getitem__(key)


class SubSpaceMixin(IndexerMixin):

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        print('set subspace loc')
        col = self.obj._get_col(key)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace'):
                raise SuperSpaceIndexError(self.name, col)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        print('get subspace loc')
        col = self.obj._get_col(key)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace'):
                self.obj.set_frame(col)
        return super().__getitem__(key)


class Indexer:
    """Extend pandas Indexer methods."""

    def __init__(self, *mixin):
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