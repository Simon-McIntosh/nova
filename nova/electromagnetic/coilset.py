"""Construct coilset with frameset and biot factories."""
from dataclasses import dataclass

from nova.electromagnetic.framefactory import FrameFactory
from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.biotfactory import BiotFactory
from nova.geometry.polygon import Polygon


@dataclass
class CoilSet(BiotFactory, FrameFactory):
    """
    Manage coilset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """

    def __post_init__(self):
        """Set filepath."""
        self.set_path('data/Nova')
        super().__post_init__()

    def __add__(self, other: FrameSet):
        """Return framset union of self and other."""
        frame = self.frame + other.frame
        subframe = self.subframe + other.subframe
        coilset = CoilSet()
        coilset.frames = frame, subframe
        return coilset

    def __iadd__(self, other: FrameSet):
        """Return coilset augmented by other."""
        self.clear_biot()
        self.frame += other.frame
        self.subframe += other.subframe
        return self


if __name__ == '__main__':

    filename = 'biot'
    reload = False
    if reload:
        coilset = CoilSet(dcoil=-35, dplasma=-500)
        coilset.coil.insert(1, 0.5, 0.95, 0.95, section='hex', turn='r',
                            nturn=-0.8)
        coilset.coil.insert(1, -0.5, 0.95, 0.95, section='hex', turn='c',
                            tile=True, delta=-6, name='bubble')
        coilset.coil.insert(2, 0, 0.95, 0.1, section='sk', nturn=-1.8)
        coilset.coil.insert(3, 0, 0.6, 0.9, section='r', turn='sk')
        coilset.plasma.insert({'ellip': [4.2, -0.4, 1.25, 4.2]}, turn='hex')
        coilset.shell.insert({'e': [2.5, -1.25, 1.75, 1.0]}, 13, 0.05,
                             delta=-40, part='vv')

        coilset.sloc['Ic'] = 1
        coilset.sloc['Shl0', 'Ic'] = -5

        coilset.grid.solve(500, 0.1, 'plasma')
        coilset.plasmagrid.solve()

        coilset.sloc['Ic'] = 6
        coilset.sloc['bubble', 'Ic'] = 5
        coilset.sloc['Shl0', 'Ic'] = -5
        coilset.sloc['plasma', 'Ic'] = -1
        coilset.store(filename)
    else:
        coilset = CoilSet().load(filename)

    separatrix = Polygon(dict(c=[4.5, 0.25, 0.9])).boundary
    coilset.plasma.update_separatrix(separatrix)

    coilset.sloc['bubble', 'Ic'] = 8
    coilset.sloc['passive', 'Ic'] = 4
    coilset.sloc['plasma', 'Ic'] = 1

    coilset.plot()
    coilset.plasma.plot()

    coilset.grid.plot(levels=31)
    coilset.plasmagrid.plot(levels=coilset.grid.levels, colors='C6')

    coilset.plasmagrid.load_operators(10)
    coilset.plasmagrid.plot_svd(levels=coilset.grid.levels)
