"""Biot-Savart calculation for complete circular cylinders."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import dask.array as da
import numpy as np

from nova.electromagnetic.biotconstants import BiotConstants
from nova.electromagnetic.biotmatrix import BiotMatrix



class BiotPolygon(BiotMatrix):
    """
    Extend Biotmatrix base class.

    Compute interaction for complete toroidal coils with polygonal sections.

    """

    _edge: int = field(init=False, default=0, repr=False)

    name: ClassVar[str] = 'polygon'  # element name
    attrs: ClassVar[list[str]] = dict(area='area', r='x', z='z')

    def __post_init__(self):
        """Extract polygon edges."""
        super().__post_init__()
        rs = [list(poly.poly.boundary.xy[0]) for poly in self.source.poly]
        print(rs)



if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=-1, dplasma=-150)
    coilset.coil.insert([5, 6], 0.5, 0.2, 0.2, section='h', turn='r',
                        nturn=300, segment='polygon')
    coilset.grid.solve(100)
    coilset.plot()
