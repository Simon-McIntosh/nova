"""Build interaction matrix for a set of poloidal points."""
from dataclasses import dataclass, field
import os

from nova.definitions import root_dir
from nova.electromagnetic.biotfilament import Biot
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.coilset import CoilSet


@dataclass
class BiotPoint(BiotSolve):
    """Compute interaction across grid."""

    def solve(self, points):
        """Solve Biot interaction across grid."""
        target = dict(x=points[0], z=points[1])
        self.data = Biot(self.subframe, target, reduce=[True, False],
                         columns=['Psi']).data
        # insert grid data
        self.data.coords['x'] = target['x']
        self.data.coords['z'] = target['z']


if __name__ == '__main__':

    file = 'MD_UP_exp22ms'
    coilset = CoilSet(file=os.path.join(root_dir,
                                        f'data/NOVA/coilsets/{file}.h5'))
    point = BiotPoint(coilset.frame, coilset.subframe, 'point')
    point.solve([[12, 13], [0, 1]])

    coilset.plot()
