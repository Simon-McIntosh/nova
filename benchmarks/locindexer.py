"""Benchmark frameset loc indexer methods."""
import os
import timeit

from nova.electromagnetic.coilset import CoilSet


class CoilSetCache:
    """Cache reference frameset."""

    number = 5000
    timer = timeit.default_timer
    filename = './frameset'

    def setup_cache(self):
        """Build reference coilset."""
        coilset = CoilSet(dcoil=-100, dplasma=-500,
                          array=['Ic', 'nturn', 'fix', 'free', 'plasma'])
        coilset.coil.insert(5.5, [-2, -1, 1], 0.5, 0.75, label='PF', free=True)
        coilset.coil.insert(4.5, [-3.5, 2.5], 0.5, 0.75, label='PF')
        coilset.coil.insert(3, range(-3, 3), 0.4, 0.9, label='CS', free=True)
        coilset.linkframe(['CS2', 'CS3'])
        coilset.firstwall.insert({'ellip': [4.2, -0.4, 1.25, 3.2]}, turn='hex')
        coilset.saloc['Ic'] = range(len(coilset.sloc))
        coilset.store(self.filename)

    def setup(self):
        """Load coilset from file."""
        self.coilset = CoilSet().load(self.filename)

    def remove(self):
        """Remove coilset."""
        os.remove(self.filename + '.nc')


class GetSubspace(CoilSetCache):
    """Benchmark subspace locindexer getters."""

    def time_subframe_subspace_getitem(self):
        """Time current access on subspace getitem."""
        return self.coilset.subframe.subspace['Ic']

    def time_subframe_subspace_geattr(self):
        """Time current access on subspace getitem."""
        return self.coilset.subframe.subspace.Ic

    def time_sloc(self):
        """Time current access via sloc indexer."""
        return self.coilset.sloc['Ic']

    def time_saloc(self):
        """Time current access via saloc indexer."""
        return self.coilset.saloc['Ic']

    def time_sloc_free(self):
        """Time (free) current access via sloc indexer."""
        return self.coilset.sloc['free', 'Ic']

    def time_saloc_free(self):
        """Time (free) current access via chained sloc indexer."""
        return self.coilset.saloc['Ic'][self.coilset.saloc['free']]


class GetSubFrame(CoilSetCache):
    """Benchmark subframe locindexer getters."""

    def time_loc(self):
        """Time nturn access via loc indexer."""
        return self.coilset.aloc['nturn']


if __name__ == '__main__':

    frameset = GetSubspace()
    frameset.setup_cache()
    frameset.setup()
    frameset.coilset.plot()
    frameset.remove()
