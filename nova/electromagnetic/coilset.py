"""Build coilset."""
from dataclasses import dataclass

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.shell import Shell



@dataclass
class Section:
    """Manage sectional properties."""

    section: str = 'rectangle'
    turn: str = 'circle'
    turn_fraction: float = 1

    def __post_init__(self):
        metadata = {'section': self.section, 'turn': self.turn,
                    'turn_fraction': self.turn_fraction}
        metadata |= self.metadata


@dataclass
class CoilSet(FrameSet):
    """
    Manage coilset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """
    dcoil: float = -1
    dplasma: float = 0.25
    dshell: float = 2.5
    dsubshell: float = 0.25
    dfield: float = 0.2

    def __post_init__(self):
        """Init mesh methods."""
        super().__post_init__()
        self.coil = Coil(self.frame, self.subframe, self.dcoil)
        self.shell = Shell(self.frame, self.subframe,
                           self.dshell, self.dsubshell)


if __name__ == '__main__':

    coilset = CoilSet(dcoil=0.05)
    coilset.coil.insert(range(3), 1, 0.75, 0.75, link=True, delta=0.2,
                        section='circle')
    coilset.shell.insert([1, 2, 3], [3, 4, 4], dt=0.1)
    coilset.subframe.polyplot()
