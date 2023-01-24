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
        Coil force resoultion. The default is -100.

            - > 0: probe segment resolution
            - int <= 0: probe segment number

    """

    dforce: float = -100
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
        for name in self.Loc['coil', :].index:
            polyframe = self.frame.loc[name, 'poly']
            polygrid = PolyGrid(polyframe, turn='rectangle', delta=self.dforce,
                                nturn=self.Loc[name, 'nturn'])
            self.target.insert(polygrid.frame,
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

    def plot_points(self, axes=None, **kwargs):
        """Plot force intergration points."""
        self.get_axes(axes, '2d')
        kwargs = dict(marker='o', linestyle='', color='C2', ms=4) | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)

    def plot(self, scale=1, norm=None, axes=None, **kwargs):
        """Plot force vectors and intergration points."""
        self.get_axes(axes, '2d')
        vector = np.c_[self.fr, self.fz]
        if norm is None:
            norm = np.max(np.linalg.norm(vector, axis=1))
        length = scale * vector / norm
        patch = self.mpl['patches'].FancyArrowPatch
        if self.reduce:
            tail = np.c_[self.data.xo, self.data.zo]
        else:
            tail = np.c_[self.data.x, self.data.z]
        arrows = [patch((x, z), (x+dx, z+dz), mutation_scale=1,
                        arrowstyle='simple,head_length=0.4, head_width=0.3,'
                        ' tail_width=0.1', shrinkA=0, shrinkB=0)
                  for x, z, dx, dz in
                  zip(tail[:, 0], tail[:, 1], length[:, 0], length[:, 1])]
        collections = self.mpl.collections.PatchCollection(
            arrows, facecolor='black', edgecolor='darkgray')
        self.axes.add_collection(collections)
        return norm
