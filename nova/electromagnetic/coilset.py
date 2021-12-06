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
from nova.electromagnetic.winding import Winding
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
                                     ferritic=Ferritic, winding=Winding))
    _biot: dict[str, BiotData] = field(
        init=False, repr=False,
        default_factory=lambda: dict(grid=BiotGrid, point=BiotPoint,
                                     probe=BiotPoint, loop=BiotLoop,
                                     inductance=BiotInductance))

    def __post_init__(self):
        """Assert _frame and _biot keys are unique."""
        assert all([attr not in self._biot for attr in self._frame])
        super().__post_init__()

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
            if isinstance(biot := self._biot.get(attr), BiotData):
                biot.store(file)

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

    def plot(self, index=None, axes=None, **kwargs):
        """Plot coilset."""
        self.subframe.polyplot(index=index, axes=axes, **kwargs)


if __name__ == '__main__':

    coilset = CoilSet(dcoil=-35, dplasma=-100)
    coilset.coil.insert(1, 0.5, 0.95, 0.95, section='hex', turn='r',
                        nturn=-0.8)
    coilset.coil.insert(1, -0.5, 0.95, 0.95, section='hex', turn='c',
                        tile=True, delta=-6, name='bubble')
    coilset.coil.insert(2, 0, 0.95, 0.1, section='sk', nturn=-1.8)
    coilset.coil.insert(3, 0, 0.6, 0.9, section='r', turn='sk')
    coilset.plasma.insert({'ellip': [2.5, 1.4, 1.6, 2.2]}, turn='hex')
    coilset.shell.insert({'e': [2.5, -1.25, 1.75, 1.0]}, 13, 0.05,
                         delta=-40, part='vv')

    coilset.plasma.generate()

    coilset.sloc['Ic'] = 1
    coilset.sloc['Shl0', 'Ic'] = -5

    import numpy as np
    coilset.probe.solve(np.random.rand(50, 2))
    print(coilset.probe.data['Psi'])

    coilset.grid.solve(65**2, 0.05)

    coilset.sloc['Ic'] = range(6)
    coilset.sloc['bubble', 'Ic'] = 5
    coilset.sloc['Shl0', 'Ic'] = -5
    coilset.sloc['plasma', 'Ic'] = -2

    coilset.plasma.update({'e': [2.2, 1.4, 0.8, 1.1]})

    coilset.grid.update_turns()

    coilset.grid.plot(levels=31)
    coilset.plasma.plot()
    coilset.plot()
