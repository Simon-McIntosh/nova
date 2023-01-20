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
        self.target = BiotFrame()
        index = self.frame.index[self.Loc['coil']]
        for name in index:
            polyframe = self.frame.loc[name, 'poly']
            polygrid = PolyGrid(polyframe, turn='rectangle', delta=self.dforce,
                                nturn=self.Loc[name, 'nturn'])
            self.target.insert(polygrid.frame, part=self.loc[name, 'part'],
                               xo=self.loc[name, 'x'],
                               zo=self.loc[name, 'z'],
                               link=True, label=name, delim='_')
        self.data = BiotSolve(self.subframe, self.target,
                              reduce=[True, True], turns=[True, True],
                              attrs=['Fr', 'Fz', 'Fzdz'],
                              name=self.name).data
        # insert grid data
        self.data.coords['subref'] = 'target', self.Loc['coil', 'subref']
        self.data.coords['x'] = self.target.x
        self.data.coords['z'] = self.target.z
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot force intergration points."""
        self.get_axes(axes, '2d')
        kwargs = dict(marker='o', linestyle='', color='C2', ms=4) | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)
