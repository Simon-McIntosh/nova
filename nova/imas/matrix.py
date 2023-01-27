"""Generate and benchmark force and field coupling matricies."""
from dataclasses import dataclass

from nova.imas.database import Ids
from nova.imas.operate import Operate


@dataclass
class Matrix(Operate):
    """Calculate force and field copuling matricies."""

    pulse: int = 105028
    run: int = 1
    pf_active: Ids | bool | str = True

    def plot(self):
        """Plot coilset, fluxmap and coil force vectors."""
        super().plot()
        self.grid.plot()
        self.plasma.wall.plot()
        self.force.plot(scale=2)


if __name__ == '__main__':

    matrix = Matrix()

    matrix.itime = 200
    matrix.plot()
