"""Manage fuse equilibrium generation and reconstruction tools."""

from dataclasses import dataclass

from nova.imas.machine import Machine


@dataclass
class Fuse(Machine):
    """
    Equilibrium generation and reconstruction methods.

    Join model predictions with data observations.
    """

    ncontour: int = 10
    pf_passive: bool = False


if __name__ == '__main__':

    fuse = Fuse()

    fuse.plot()
