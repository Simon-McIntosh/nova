"""Build coilset."""
from dataclasses import dataclass, field

from nova.electromagnetic.biotgrid import BiotGrid
from nova.electromagnetic.biotinductance import BiotInductance
from nova.electromagnetic.biotloop import BiotLoop
from nova.electromagnetic.biotpoint import BiotPoint
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.frameset import Frame, FrameSet
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

    def __post_init__(self):
        """Construct delta lookup."""
        self._delta = dict(coil=self.dcoil, shell=self.dshell,
                           plasma=self.dplasma, field=self.dfield)
        super().__post_init__()


@dataclass
class CoilSet(CoilGrid, FrameSet):
    """
    Manage coilset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """

    _frame: dict[str, Frame] = field(
        init=False, repr=False,
        default_factory=lambda: dict(coil=Coil, shell=Shell, plasma=Plasma))
    _biot: dict[str, BiotSolve] = field(
        init=False, repr=False,
        default_factory=lambda: dict(grid=BiotGrid, point=BiotPoint,
                                     probe=BiotPoint, loop=BiotLoop,
                                     inductance=BiotInductance))

    def __post_init__(self):
        """Init."""
        super().__post_init__()

    def __getattr__(self, attr):
        """Intercept attribute access - implement frame methods."""
        if attr in self._frame:
            if isinstance(frame := self._frame.get(attr), Frame):
                return frame
            delta = self._delta.get(attr, self.delta)
            self._frame[attr] = frame(*self.frames, delta)
            return self._frame[attr]
        if attr in self._biot:
            if isinstance(biot := self._biot.get(attr), BiotSolve):
                return biot
            self._biot[attr] = biot(*self.frames, attr)
            return self._biot[attr]
        raise AttributeError

    def store(self, file):
        """Store coilset to hdf5 file."""
        super().store(file)
        '''
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
        '''

    def load(self, file):
        """Load coilset from hdf5 file."""
        super().load(file)
        '''
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
        '''

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
    #coilset.coil.insert(11, 0, 0.95, 0.1, section='sk', nturn=-1.8)
    #coilset.coil.insert(12, 0, 0.6, 0.9, section='r', turn='sk')
    coilset.plasma.insert({'ellip': [11.5, 0.8, 1.7, 0.4]})
    coilset.shell.insert({'e': [11, 0, 5.75, 3.25]}, 13, 0.05,
                         delta=-40, part='vv')
    coilset.link(['Coil0', 'Plasma'], 2)

    coilset.sloc['Ic'] = 1
    coilset.sloc['Shl0', 'Ic'] = -5


    print(coilset)

    coilset.plot()
    plt.plot(*coilset.subframe.loc[:, ['x', 'z']].to_numpy().T, '.')

    coilset.grid.solve(1e3, 0.05)
    coilset.grid.plot()

    coilset.store('tmp')

    #coilset.frame.vtkplot()
