"""Build coilset."""
from dataclasses import dataclass, field

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.biotgrid import BiotGrid
from nova.electromagnetic.biotinductance import BiotInductance
from nova.electromagnetic.biotpoint import BiotPoint
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.shell import Shell
from nova.electromagnetic.plasma import Plasma

from nova.utilities.pyplot import plt


@dataclass
class CoilGrid:
    """Default grid sizing parameters."""

    dcoil: float = -1
    dplasma: float = 0.25
    dshell: float = 0
    dfield: float = 0.2


@dataclass
class CoilSet(CoilGrid, FrameSet):
    """
    Manage coilset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """

    file: str = field(default=None)

    def __post_init__(self):
        """Init mesh methods."""
        super().__post_init__()
        self.coil = Coil(*self.frames, self.dcoil)
        self.shell = Shell(*self.frames, self.dshell)
        self.plasma = Plasma(*self.frames, self.dplasma)
        self.grid = BiotGrid(*self.frames, 'grid')
        self.point = BiotPoint(*self.frames, 'point')
        self.probe = BiotPoint(*self.frames, 'probe')
        #self.loop = BiotLoop(*self.frames, 'loop')
        self.inductance = BiotInductance(*self.frames, 'inductance')
        if self.file is not None:
            self.load(self.file)

    def store(self, file):
        """Store coilset to hdf5 file."""
        super().store(file)
        try:
            self.grid.store(file)
        except AttributeError:
            pass
        try:
            self.point.store(file)
        except AttributeError:
            pass
        try:
            self.probe.store(file)
        except AttributeError:
            pass
        try:
            self.loop.store(file)
        except AttributeError:
            pass
        try:
            self.inductance.store(file)
        except AttributeError:
            pass

    def load(self, file):
        """Load coilset from hdf5 file."""
        super().load(file)
        self.plasma.generate()
        try:
            self.grid.load(file)
        except OSError:
            pass
        try:
            self.point.load(file)
        except OSError:
            pass
        try:
            self.probe.load(file)
        except OSError:
            pass
        #try:
        #    self.loop.load(file)
        #except OSError:
        #    pass
        try:
            self.inductance.load(file)
        except OSError:
            pass

    def plot(self, axes=None):
        """Plot coilset."""
        self.subframe.polyplot(axes=axes)


if __name__ == '__main__':
    '''
    coilset = CoilSet(dplasma=-50)
    coilset.coil.insert(0.8, [0.5, 1, 1.5], 0.25, 0.45, link=True, delta=-10,
                        section='r', scale=0.75,
                        nturn=24, turn='c', part='pf')
    coilset.shell.insert({'e': [1.5, 1, 0.75, 1.25]}, -5, 0.05,
                         delta=-40, part='vv')
    coilset.plasma.insert({'sk': [1.5, 1, 0.5, 0.5]}, turn='hex', tile=True,
                          trim=True)
    coilset.link(['Shl0', 'Shl4', 'Shl1'])

    coilset.sloc['coil', 'Ic'] = [-12]
    coilset.sloc['plasma', 'Ic'] = [9.9]
    coilset.sloc['passive', 'Ic'] = [7.7, 12, 5.5]

    coilset.sloc[:, 'Ic'] = 9.3


    coilset.plot()

    '''

    coilset = CoilSet(dcoil=-35, dplasma=-40)
    coilset.coil.insert(10, 0.5, 0.95, 0.95, section='hex', turn='r',
                        nturn=-0.8)
    coilset.coil.insert(10, -0.5, 0.95, 0.95, section='hex')
    coilset.coil.insert(11, 0, 0.95, 0.1, section='sk', nturn=-1.8)
    coilset.coil.insert(12, 0, 0.6, 0.9, section='r', turn='sk')
    coilset.plasma.insert({'ellip': [11.5, 0.8, 1.7, 0.4]})
    coilset.shell.insert({'e': [13, 0, 0.75, 1.25]}, 13, 0.05,
                         delta=-40, part='vv')
    coilset.link(['Coil0', 'Plasma'], 2)

    coilset.sloc['Ic'] = 1

    print(coilset)

    coilset.plot()
    plt.plot(*coilset.subframe.loc[:, ['x', 'z']].to_numpy().T, '.')
    #coilset.grid.plot()
