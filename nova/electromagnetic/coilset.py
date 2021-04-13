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
    dshell: float = 2.5
    dsubshell: float = 0.25
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
                           self.dshell, self.dsubshell)
        self.plasma = Plasma(self.frame, self.subframe, self.dplasma)

    def plot(self):
        """Plot coilset."""
        self.subframe.polyplot()


if __name__ == '__main__':

    coilset = CoilSet(dplasma=0.25)

    coilset.coil.insert(range(2), 1, 0.75, 0.5, link=True, delta=-10,
                        skin=0.65, section='r',
                        nturn=24, turn='c', tile=True)
    coilset.plasma.insert({'sk': [1.7, 1, 0.5, 0.85]}, turn='hex', tile=True,
                          trim=True)

    coilset.plot()


    '''
    coilset.coil.insert(range(3), 1, 0.75, 0.75, link=True, delta=0.2,
                        section='circle')
    coilset.shell.insert([1, 2, 3], [3, 4, 4], dt=0.1)
    coilset.subframe.polyplot()
    '''
