"""Manage unstructured kd-tree."""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import box

from nova.biot.grid import Expand
from nova.biot.plasmagrid import PlasmaGrid
from nova.biot.solve import Solve
from nova.frame.polygrid import PolyGrid
from nova.geometry.kdtree import Proximate


@dataclass
class HexGrid(Proximate, PlasmaGrid):
    """Implement kd-tree querys for arbitaray unstructured grids."""

    limit: float = 0.2
    kd_factor: float = 0.1
    turn: str = 'hexagon'
    tile: bool = True

    def solve(self, number=None, limit=None, index='plasma'):
        """Overwrid PlasmaGrid.solve to permit arbitrary solution domains."""
        if limit is None:
            limit = self.limit
        with self.solve_biot(number) as number:
            if number is not None:
                limit = Expand(self.subframe, index)(limit)
                target = PolyGrid(limit, delta=-number,
                                  turn=self.turn, tile=self.tile).frame
                wall = box(*limit[::2], *limit[1::2]).boundary
                self.data = Solve(self.subframe, target, reduce=[True, False],
                                  attrs=self.attrs, name=self.name).data
                self.tessellate(target, wall)

    def load_operators(self):
        """Extend Grid.load_operators to update tree instance."""
        super().load_operators()
        if self.number is not None:
            self.kd_points = np.c_[self.data.x.data, self.data.z.data]
