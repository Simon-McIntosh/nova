"""Solve intergral coil forces."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.biotframe import BiotFrame
from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.frame.baseplot import Plot
from nova.frame.polygrid import PolyGrid


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
    reduce: bool = True
    attrs: list[str] = field(default_factory=lambda: ['Fr', 'Fz', 'Fc'])
    target: BiotFrame = field(init=False, repr=False)

    def __len__(self):
        """Return force patch number."""
        return len(self.data.get('x', []))

    def solve(self, dforce=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        if dforce is not None:
            self.dforce = dforce
        self.target = BiotFrame()
        for name in self.frame.index[self.Loc['coil']]:
            polyframe = self.frame.loc[name, 'poly']
            polygrid = PolyGrid(polyframe, turn='rectangle', delta=self.dforce,
                                nturn=self.Loc[name, 'nturn'])
            self.target.insert(polygrid.frame, part=self.loc[name, 'part'],
                               xo=self.loc[name, 'x'],
                               zo=self.loc[name, 'z'],
                               link=True, label=name, delim='_')
        self.data = BiotSolve(self.subframe, self.target,
                              reduce=[True, self.reduce], turns=[True, True],
                              attrs=self.attrs, name=self.name).data
        # insert grid data
        self.data.coords['index'] = 'target', self.Loc['coil', 'subref']
        if self.reduce:
            self.data.coords['xo'] = 'target', self.Loc['coil', 'x']
            self.data.coords['zo'] = 'target', self.Loc['coil', 'z']
            self.data.coords['x'] = self.target.x
            self.data.coords['z'] = self.target.z
        else:
            self.data.coords['x'] = 'target', self.target.x
            self.data.coords['z'] = 'target', self.target.z
        super().post_solve()

    def plot(self, scale=0.5, axes=None, **kwargs):
        """Plot force vectors and intergration points."""
        self.get_axes(axes, '2d')
        kwargs = dict(marker='o', linestyle='', color='C2', ms=4) | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)
        vector = np.c_[self.fr, self.fz]
        norm = np.linalg.norm(vector, axis=1)
        length = scale * vector / norm[:, np.newaxis]
        patch = self.mpl['patches'].FancyArrowPatch
        if self.reduce:
            tail = np.c_[self.data.xo, self.data.zo]
        else:
            tail = np.c_[self.data.x, self.data.z]
        arrows = [patch((x, z), (x+dx, z+dz), mutation_scale=0.2*scale)
                  for x, z, dx, dz in
                  zip(tail[:, 0], tail[:, 1], length[:, 0], length[:, 1])
                  if np.isfinite((dx, dz)).all()]
        collections = self.mpl.collections.PatchCollection(arrows)
        self.axes.add_collection(collections)
