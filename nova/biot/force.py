"""Solve intergral coil forces."""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from nova.biot.biotframe import BiotFrame
from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.frame.baseplot import Plot
from nova.geometry.polyframe import PolyFrame



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

    def solve(self, dforce=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        if dforce is not None:
            self.dforce = dforce
        self.target = BiotFrame(label='Force')
        index = []
        for name in self.loc['coil', 'frame'].unique():

            polyframe = self.extract_polyframe(name)
            if polyframe.poly.boundary.is_ring:
                sample = Sample(polyframe.boundary, delta=self.dfield)
                self.target.insert(sample['radius'], sample['height'],
                                   link=True)
                index.append(name)
        self.data = BiotSolve(self.subframe, self.target,
                              reduce=[True, False], turns=[True, False],
                              attrs=['Br', 'Bz'], name=self.name).data
        # insert grid data
        self.data.coords['index'] = index
        self.data.coords['x'] = self.target.x
        self.data.coords['z'] = self.target.z
        super().post_solve()
