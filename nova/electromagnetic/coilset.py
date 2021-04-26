"""Build coilset."""
from dataclasses import dataclass, field

from nova.electromagnetic.framepack import FramePack
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.shell import Shell
from nova.electromagnetic.plasma import Plasma


@dataclass
class FrameGrid:
    """Default grid sizing parameters."""

    dcoil: float = -1
    dplasma: float = 0.25
    dshell: float = 0
    dfield: float = 0.2


@dataclass
class CoilSet(FrameGrid, FramePack):
    """
    Manage coilset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """

    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Init mesh methods."""
        super().__post_init__()
        self.coil = Coil(self.frame, self.subframe, self.dcoil)
        self.shell = Shell(self.frame, self.subframe, self.dshell)
        self.plasma = Plasma(self.frame, self.subframe, self.dplasma)

    def plot(self):
        """Plot coilset."""
        self.subframe.polyplot()


if __name__ == '__main__':

    coilset = CoilSet(dplasma=-50)

    coilset.coil.insert(0.8, [0.5, 1, 1.5], 0.25, 0.45, link=True, delta=-10,
                        section='r', scale=0.75,
                        nturn=24, turn='c', part='pf')
    coilset.shell.insert({'e': [1.5, 1, 0.75, 1.25]}, -5, 0.05,
                         delta=-40, part='vv')
    coilset.plasma.insert({'sk': [1.5, 1, 0.5, 0.5]}, turn='hex', tile=True,
                          trim=True)

    coilset.link(['Shl0', 'Shl4', 'Shl1'])

    coilset.plot()

    coilset.sloc['coil', 'Ic'] = [-12]
    coilset.sloc['plasma', 'Ic'] = [9.9]
    coilset.sloc['passive', 'Ic'] = [7.7, 12, 5.5]

    coilset.sloc[:, 'Ic'] = 9.3


    def set_current():
        coilset.loc['Ic'] = 6

    def get_current():
        _ = coilset.sloc['active', 'Ic']

