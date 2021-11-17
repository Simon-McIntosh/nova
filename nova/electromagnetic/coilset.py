"""Build coilset."""
from dataclasses import dataclass, field

from nova.electromagnetic.biotgrid import BiotGrid
from nova.electromagnetic.biotinductance import BiotInductance
from nova.electromagnetic.biotloop import BiotLoop
from nova.electromagnetic.biotpoint import BiotPoint
from nova.electromagnetic.biotdata import BiotData
from nova.electromagnetic.coil import Coil
from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.framedata import FrameData
from nova.electromagnetic.shell import Shell
from nova.electromagnetic.plasma import Plasma
from nova.electromagnetic.ferritic import Ferritic


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

    _frame: dict[str, FrameData] = field(
        init=False, repr=False,
        default_factory=lambda: dict(coil=Coil, shell=Shell, plasma=Plasma,
                                     ferritic=Ferritic))
    _biot: dict[str, BiotData] = field(
        init=False, repr=False,
        default_factory=lambda: dict(grid=BiotGrid, point=BiotPoint,
                                     probe=BiotPoint, loop=BiotLoop,
                                     inductance=BiotInductance))

    def __getattr__(self, attr):
        """Intercept attribute access - implement frame methods."""
        if attr in self._frame:
            if isinstance(frame := self._frame.get(attr), FrameData):
                return frame
            delta = self._delta.get(attr, self.delta)
            self._frame[attr] = frame(*self.frames, delta=delta)
            return self._frame[attr]
        if attr in self._biot:
            if isinstance(biot := self._biot.get(attr), BiotData):
                return biot
            self._biot[attr] = biot(*self.frames, name=attr)
            return self._biot[attr]
        raise AttributeError

    def store(self, file):
        """Store coilset to hdf5 file."""
        super().store(file)
        for attr in self._biot:
            try:
                getattr(self, attr).store(file)
            except AttributeError:
                pass

    def load(self, file):
        """Load coilset from hdf5 file."""
        super().load(file)
        self.plasma.generate()
        for attr in self._biot:
            try:
                getattr(self, attr).load(file)
            except OSError:
                pass
        return self

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
    '''
    coilset.coil.insert(10, 0.5, 0.95, 0.95, section='hex', turn='r',
                        nturn=-0.8)
    coilset.coil.insert(10, -0.5, 0.95, 0.95, section='hex')
    coilset.coil.insert(11, 0, 0.95, 0.1, section='sk', nturn=-1.8)
    coilset.coil.insert(12, 0, 0.6, 0.9, section='r', turn='sk')
    coilset.plasma.insert({'ellip': [11.5, 0.8, 1.7, 0.4]})
    coilset.shell.insert({'e': [12, -0.75, 1.75, 0.5]}, 13, 0.05,
                         delta=-40, part='vv')
    coilset.link(['Coil0', 'Plasma'], 2)

    coilset.sloc['Ic'] = 1
    coilset.sloc['Shl0', 'Ic'] = -5

    #coilset.grid.solve(1e3, 0.05)
    #coilset.grid.plot()

    #ferriticframe = ShieldCluster().frame
    '''

    #from nova.electromagnetic.ferritic import ShieldSet
    #frame = ShieldSet('IWS_CFM').frame
    #frame.polyplot()

    coilset = CoilSet()



    #import vedo
    #vedo.show(coilset.frame.vtk)

    #coilset.frame.vtkplot()
