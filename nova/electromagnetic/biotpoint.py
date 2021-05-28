"""Build interaction matrix for a set of poloidal points."""
from dataclasses import dataclass, field
import os

from nova.definitions import root_dir
from nova.electromagnetic.biotfilament import Biot
from nova.electromagnetic.biotdata import BiotMatrix
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotPoint(BiotSolve, FrameSetLoc):
    """Compute interaction across grid."""

    frame: FrameSpace = field(repr=False)
    subframe: FrameSpace = field(repr=False)
    data: BiotMatrix = field(init=False, repr=False)

    def solve(self, points):
        """Solve Biot interaction across grid."""
        target = dict(x=points[0], z=points[1])
        self.data = Biot(self.subframe, target, reduce=[True, False],
                         columns=['Psi']).data
        # insert grid data
        self.data.coords['x'] = target['x']
        self.data.coords['z'] = target['z']


if __name__ == '__main__':

    file = 'ITER.h5'
    coilset = CoilSet(file=os.path.join(root_dir,
                                        f'data/NOVA/coilsets/{file}'))
    point = BiotPoint(frame=coilset.frame, subframe=coilset.subframe,
                      name='point')
    point.solve([[12, 13], [0, 1]])

    #coilset.plot()
