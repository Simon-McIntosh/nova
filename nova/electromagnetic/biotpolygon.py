"""Biot-Savart calculation for complete circular cylinders."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import dask.array as da
import numpy as np

from nova.electromagnetic.biotconstants import BiotConstants
from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotmatrix import BiotMatrix


class BiotPolygon(BiotMatrix):
    """
    Extend Biotmatrix base class.

    Compute interaction for complete toroidal coils with polygonal sections.

    """

    _edge: int = field(init=False, default=0, repr=False)

    name: ClassVar[str] = 'polygon'  # element name
    attrs: ClassVar[list[str]] = dict(area='area', r='x', z='z')
    metadata: ClassVar[dict[str, str]] = dict(
        required=['ref', 'r1', 'z1', 'r2', 'z2'], additional=[],
        available=[], array=[], base=[])

    def __post_init__(self):
        """Extract polygon edges."""
        super().__post_init__()

        edge = BiotFrame(**self.metadata, label='edge', delim='-')
        for i, poly in enumerate(self.source.poly):
            coords = poly.poly.boundary.xy
            edge.insert(i,
                        coords[0][:-1], coords[0][1:],
                        coords[1][:-1], coords[1][1:],
                        metadata=self.metadata)
        edge.set_target(len(self.target))
        print(len(self.target))
        print(edge('r1'))
        #self.edge = BiotFrame(required=['r1', 'z1', 'r2', 'z2'])




if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=-1, dplasma=-150)
    coilset.coil.insert([5, 6], 0.5, 0.2, 0.2, section='h', turn='r',
                        nturn=300, segment='polygon')
    coilset.grid.solve(100)
    coilset.plot()
