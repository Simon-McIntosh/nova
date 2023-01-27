"""Generate and benchmark force and field coupling matricies."""
from dataclasses import dataclass

from nova.imas.database import Ids
from nova.imas.operate import Operate


@dataclass
class Matrix(Operate):
    """Calculate force and field copuling matricies."""

    pulse: int = 105028
    run: int = 1
    time_index: int = 20
    pf_active: Ids | bool | str = 'iter_md'


if __name__ == '__main__':

    matrix = Matrix()

    # operate.itime = 200
