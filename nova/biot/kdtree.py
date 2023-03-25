"""Manage unstructured kd-tree."""
from dataclasses import dataclass

from shapely.geometry import box

from nova.biot.grid import Expand
from nova.biot.plasmagrid import PlasmaGrid
from nova.biot.solve import Solve
from nova.frame.polygrid import PolyGrid


@dataclass
class KDTree(PlasmaGrid):
    """Implement kd-tree querys for arbitaray unstructured grids."""

    turn: str = 'hexagon'
    tile: bool = True

    def solve(self, number=None, limit=0.2, index='plasma'):
        """Overwrid PlasmaGrid.solve to permit arbitrary solution domains."""
        with self.solve_biot(number) as number:
            if number is not None:
                limit = Expand(self.subframe, index)(limit)
                target = PolyGrid(limit, delta=-number,
                                  turn=self.turn, tile=self.tile).frame
                wall = box(*limit[::2], *limit[1::2]).boundary
                self.data = Solve(self.subframe, target, reduce=[True, False],
                                  attrs=self.attrs, name=self.name).data
                self.tessellate(target, wall)
