"""Manage fuse equilibrium generation and reconstruction tools."""

from dataclasses import dataclass

from nova.imas.database import Ids
from nova.imas.machine import Machine


@dataclass
class Fuse(Machine):
    """
    EquilibriumData generation and reconstruction methods.

    Merge model predictions with data observations.
    """

    ncontour: int = 10
    pf_passive: Ids | bool | str = False


if __name__ == '__main__':

    fuse = Fuse()

    fuse.plot()
