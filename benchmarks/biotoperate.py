"""Benchmark biotoperate."""
import os
import timeit

from nova.electromagnetic.coilset import CoilSet


class PlasmaGrid:
    """Benchmark biotoperate methods."""

    number = 500
    params = [10, 75, 200, 500, 1500, -1]
    param_names = ['svd_rank']
    timer = timeit.default_timer

    @property
    def filename(self):
        """Return coilset filename."""
        return f'./{self.__class__.__name__.lower()}_coilset'

    def setup_cache(self):
        """Build reference coilset."""
        coilset = CoilSet(dplasma=-1500)
        coilset.firstwall.insert({'ellip': [4.2, -0.4, 1.25, 4.2]}, turn='hex')
        coilset.plasmagrid.solve()
        coilset.store(self.filename)

    def remove(self):
        """Remove coilset."""
        os.remove(self.filename + '.nc')

    def setup(self, svd_rank):
        """Load coilset from file."""
        self.coilset = CoilSet().load(self.filename)
        self.coilset.plasmagrid.svd_rank = svd_rank

    def time_update_turns(self, svd_rank):
        """Time generation of plasma grid."""
        self.coilset.plasmagrid.update_turns('Psi', svd_rank != -1)

    def time_flux_function(self, svd_rank):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.psi

    def time_radial_field(self, svd_rank):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.br

    def time_field_magnitude(self, svd_rank):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.bn


if __name__ == '__main__':

    biot = PlasmaGrid()
    biot.setup_cache()
    biot.setup(75)
    biot.coilset.plot()
    biot.time_update_turns(75)
    biot.remove()
