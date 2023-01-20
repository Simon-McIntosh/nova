"""Solve intergral coil forces."""
from dataclasses import dataclass, field

from nova.biot.biotframe import BiotFrame
from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.frame.baseplot import Plot
from nova.frame.dataframe import DataFrame
from nova.frame.polygrid import PolyGrid
from nova.frame.polyplot import PolyPlot


@dataclass
class Force(Plot, BiotOperate):
    """
    Compute coil force interaction matricies.

    Parameters
    ----------
    dforce : int | +float, optional
        Coil force resoultion. The default is -10.

            - > 0: probe segment resolution
            - int <= 0: probe segment number

    """

    dforce: float = -10
    target: BiotFrame = field(init=False, repr=False)

    def __len__(self):
        """Return force patch number."""
        return len(self.data.get('x', []))

    def solve(self, dforce=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        if dforce is not None:
            self.dforce = dforce
        self.target = BiotFrame(label='Force')
        index = []
        for name in self.loc['coil', 'frame'].unique():
            polyframe = self.frame.loc[name, 'poly']
            polygrid = PolyGrid(polyframe, turn='rectangle', delta=self.dforce,
                                nturn=self.Loc[name, 'nturn'])
            self.target.insert(polygrid.frame, part=self.loc[name, 'part'],
                               link=True)
        self.data = BiotSolve(self.subframe, self.target,
                              reduce=[True, True], turns=[True, True],
                              attrs=['Br', 'Bz', 'Brdz', 'Bzdr', 'Psi'],
                              name=self.name).data
        # insert grid data
        self.data.coords['index'] = index
        self.data.coords['x'] = self.target.x
        self.data.coords['z'] = self.target.z
        super().post_solve()

    @property
    def polyplot(self):
        """Return polyplot instance."""
        target = self.target.copy()
        return PolyPlot(DataFrame(target))

    def plot(self, axes=None):
        """Plot force polycells."""
        self.get_axes(axes, '2d')
        self.polyplot.plot()
