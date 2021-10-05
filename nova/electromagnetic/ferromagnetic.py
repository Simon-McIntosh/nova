"""Mesh feritic inserts."""
from dataclasses import dataclass, field

import numpy as np

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.vtkplot import VtkPlot

from nova.electromagnetic.coil import Coil

@dataclass
class FerroMagnetic:
    """Manage ferritic inserts."""

    frame: FrameSpace = field(repr=False)
    subframe: FrameSpace = field(repr=False)
    delta: float

    def __post_init__(self):
        """fegegw"""
        ##self.frame.metaframe.metadata = \
        #    dict(additional=['volume'])

    def insert(self, x, y, z, iloc=None, **additional):
        """
        Add ferromagnetic block to frameset.

        Block centroid described by x, y, z coordinates
        meshed into n coils based on
        dblock. Each frame is meshed based on delta.

        Parameters
        ----------
        x : array-like, shape(n,)
            x-coordinates of block cenroids.
        y : array-like, shape(n,)
            y-coordinates of block cenroids.
        z : array-like, shape(n,)
            z-coordinates of of block cenroids.

        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        additional |= dict(x=x, y=y, z=z)
        try:
            additional['volume'] = np.prod([additional[attr]
                                            for attr in ['dx', 'dy', 'dz']])
        except KeyError:
            pass
        self.frame.insert(iloc=iloc, segment='volume', **additional)


if __name__ == '__main__':

    fset = FrameSet()

    #fmag = FerroMagnetic(*fset.frames, 0.5)
    #fmag.insert(1, 1, 2, volume=0.2)
    #fmag.insert(2, 1, 2, volume=0.1)
    #fmag.insert(1, 3, 2, volume=0.5)

    coil = Coil(*fset.frames, 0.5)
    coil.insert(1, 5, dl=0.3, dt=0.1)

    fset.frame.vtkplot()


    print(fset.frame.columns)
