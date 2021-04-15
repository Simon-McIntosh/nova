"""Build coilset."""
from dataclasses import dataclass

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.shell import Shell
from nova.electromagnetic.plasma import Plasma


@dataclass
class FrameGrid:
    """Default grid sizing parameters."""

    dcoil: float = -1
    dplasma: float = 0.25
    dshell: float = 0.2
    dfield: float = 0.2


@dataclass
class CoilSet(FrameGrid, FrameSet):
    """
    Manage coilset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """

    def __post_init__(self):
        """Init mesh methods."""
        super().__post_init__()
        self.coil = Coil(self.frame, self.subframe, self.dcoil)
        self.shell = Shell(self.frame, self.subframe,
                           dshell=self.dshell, delta=self.dsubshell)
        self.plasma = Plasma(self.frame, self.subframe, self.dplasma)

    def plot(self):
        """Plot coilset."""
        self.subframe.polyplot()


if __name__ == '__main__':

    coilset = CoilSet(dplasma=0.25)

    coilset.coil.insert(range(2), 1, 0.75, 0.5, link=True, delta=-10,
                        skin=0.65, section='r', scale=0.75,
                        nturn=24, turn='hex', part='PF')
    coilset.shell.insert([0, 1.1, 2], [2, 1.4, 1.7], dt=0.1)
    coilset.plasma.insert({'sk': [1.7, 1, 0.5, 0.85]}, turn='hex', tile=True,
                          trim=True)
    coilset.plot()
