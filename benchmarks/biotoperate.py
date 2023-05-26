"""Benchmark biotoperate."""
import os
import timeit

from nova.frame.coilset import CoilSet


class PlasmaGrid:
    """Benchmark biotoperate methods - plasmagrid base class."""

    timer = timeit.default_timer

    @property
    def filename(self):
        """Return coilset filename."""
        return './plasmagrid_coilset'

    def setup_cache(self):
        """Build reference coilset."""
        coilset = CoilSet(dplasma=-500, filename=self.filename)
        coilset.firstwall.insert({'ellip': [4.2, -0.4, 1.25, 4.2]}, turn='hex')
        coilset.plasmagrid.solve()
        coilset.plasmagrid.svd_rank = 75
        coilset.store()

    def remove(self):
        """Remove coilset."""
        os.remove(self.filename + '.nc')

    def setup(self):
        """Load coilset from file."""
        self.coilset = CoilSet().load()


class PlasmaTurns(PlasmaGrid):
    """Benchmark biotoperate methods."""

    number = 5000
    params = [10, 75, 200, 500, -1]
    param_names = ['svd_rank']

    def setup(self, svd_rank):
        """Load coilset from file and set svd rank."""
        self.coilset = CoilSet(filename=self.filename).load()
        self.coilset.plasmagrid.svd_rank = svd_rank

    def time_update_turns(self, svd_rank):
        """Time generation of plasma grid."""
        self.coilset.plasmagrid.update_turns('Psi', svd_rank != -1)


class PlasmaEvaluate(PlasmaGrid):
    """Time evaluation of plasma operators."""

    number = 5000

    def time_flux_function_ev_only(self):
        """Time forced evaluation of flux function."""
        self.coilset.plasmagrid.operator['Psi'].evaluate()

    def time_flux_function(self):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.psi

    def time_radial_field(self):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.br

    def time_field_magnitude(self):
        """Time computation of radial field."""
        return self.coilset.plasmagrid.bn


class PlasmaOperate(PlasmaGrid):
    """Time plasma grid operations."""

    def time_solve(self):
        """Time plasma grid biot solution."""
        self.coilset.plasmagrid.solve()

    def time_load_operators(self):
        """Time biot operator reload."""
        self.coilset.plasmagrid.load_operators()


PlasmaOperate.time_solve.number = 10
PlasmaOperate.time_load_operators.number = 100


if __name__ == '__main__':

    biot = PlasmaTurns()
    biot.setup_cache()
    biot.setup(75)
    biot.coilset.plot()
    biot.time_update_turns(75)
    biot.remove()
