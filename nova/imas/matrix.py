"""Generate and benchmark force and field coupling matricies."""
from dataclasses import dataclass, field

from nova.imas.database import Ids
from nova.imas.operate import Operate
from nova.imas.profile import Profile


@dataclass
class Matrix(Operate):
    """Calculate force and field copuling matricies + write to file."""

    pulse: int = 135014  # 105028
    run: int = 1  # 1
    pf_active: Ids | bool | str = 'iter_md'

    def plot(self):
        """Plot coilset, fluxmap and coil force vectors."""
        super().plot()
        self.grid.plot()
        self.plasma.wall.plot()
        self.force.plot(scale=2)


@dataclass
class Benchmark(Profile):
    """Benchmark EM coupling matricies with other IDS."""

    pulse: int = 135007
    run: int = 4
    matrix: Matrix = field(default_factory=Matrix)

    '''
    def benchmark(self, ):
        pf_active = PF_Active(pulse, run)

        data_index = [i for i, name in
                      enumerate(pf_active.data.coil_name.values)
                      if name in self.sloc.frame.index]
        sloc_index = [self.sloc.frame.index.get_loc(name)
                       for name in pf_active.data.coil_name[data_index].values
                       if name in self.sloc.frame.index]


        # plasma = operate.aloc['plasma']
        # operate.aloc['nturn'][plasma] = nturn
        # operate.update_aloc_hash('nturn')

        operate.itime = 1000

        operate.plot()
        operate.grid.plot()
        operate.plasma.wall.plot()

        norm = operate.force.plot(scale=2)

        operate.set_axes(None, '1d')
        operate.axes.bar(operate.Loc['coil', :].index.values,
                         operate.force.fr*1e-6)
        operate.axes.bar(operate.Loc['coil', :].index.values,
                         operate.data.radial_force[operate.itime]*1e-6, width=0.6)

        operate.set_axes(None, '1d')
        operate.axes.bar(operate.Loc['coil', :].index.values,
                         operate.force.fz*1e-6)
        operate.axes.bar(operate.Loc['coil', :].index.values,
                         operate.data.vertical_force[operate.itime]*1e-6, width=0.6)
    '''



if __name__ == '__main__':

    benchmark = Benchmark()

    #matrix = Matrix()

    #matrix.itime = 300
    #matrix.plot()
