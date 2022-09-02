"""Construct coilset with frameset and biot factories."""
from dataclasses import dataclass

from nova.electromagnetic.framefactory import FrameFactory
from nova.electromagnetic.frameset import FrameSet
from nova.biot.biotfactory import BiotFactory
from nova.geometry.polygon import Polygon


@dataclass(repr=False)
class CoilSet(BiotFactory, FrameFactory):
    """
    Manage coilset.

    - poloidal: add poloidal coils.
    - shell: add poloidal shells.
    - plasma: add plasma (poloidal).

    """

    datapath: str = 'data/Nova'

    def __post_init__(self):
        """Set filepath."""
        self.set_path(self.datapath)
        super().__post_init__()

    @property
    def coilset_attrs(self):
        """Return coilset attrs."""
        return self.frame_attrs | self.biot_attrs

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
    reload = True
    if reload:
        coilset = CoilSet(dcoil=-5, dplasma=-150)
        coilset.coil.insert(1, 0.5, 0.95, 0.95, section='hex', turn='r',
                            nturn=-5.8, delta=-1)
        coilset.coil.insert(1, -0.5, 0.95, 0.95, section='hex', turn='c',
                            tile=True, delta=-6, name='bubble')
        coilset.coil.insert(2, 0, 0.95, 0.1, section='sk', nturn=-1.8)
        coilset.coil.insert(3, 0, 0.6, 0.9, section='r', turn='sk')
        coilset.firstwall.insert({'ellip': [4.2, -0.4, 1.25, 4.2]}, turn='hex')
        coilset.shell.insert({'e': [2.5, -1.25, 1.75, 1.0]}, 13, 0.05,
                             delta=-4, part='vv')

        coilset.sloc['Ic'] = 1
        coilset.sloc['Shl0', 'Ic'] = -5

        coilset.grid.solve(500, 0.1)  # , 'plasma'
        coilset.plasmagrid.solve()

        coilset.plasma.separatrix = dict(c=[4.5, 0.25, 0.9])

        coilset.sloc['Ic'] = 6
        coilset.sloc['bubble', 'Ic'] = 5
        coilset.sloc['Shl0', 'Ic'] = -5
        coilset.sloc['plasma', 'Ic'] = -1
        # coilset.store(filename)
    else:
        coilset = CoilSet().load(filename)

    separatrix = Polygon(dict(c=[4.5, -0.75, 0.9])).boundary
    coilset.plasma.separatrix = separatrix

    coilset.sloc['bubble', 'Ic'] = 8
    coilset.sloc['passive', 'Ic'] = 4
    coilset.sloc['plasma', 'Ic'] = 1

    coilset.plot()
    coilset.plasma.plot()

    coilset.grid.plot(levels=51)
    coilset.plasmagrid.plot(levels=coilset.grid.levels, colors='C6')

    coilset.plasmagrid.svd_rank = 10
    coilset.plasmagrid.plot_svd(levels=coilset.grid.levels)
