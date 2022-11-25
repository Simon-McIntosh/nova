"""Mesh feritic inserts."""
from dataclasses import dataclass, field

import numpy as np

from nova.frame.frameset import Frames
from nova.frame.frameset import FrameSet

from nova.frame.coil import Coil


@dataclass
class FerroMagnetic(Frames):
    """Manage ferritic inserts."""

    required: list[str] = field(
        default_factory=lambda: ['x', 'y', 'z', 'volume'])

    def insert(self, *args, required=None, iloc=None, **additional):
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
        try:
            additional['volume'] = np.prod([additional[attr]
                                            for attr in ['dx', 'dy', 'dz']])
        except KeyError:
            pass
        with self.insert_required(required):
            self.frame.insert(*args, iloc=iloc, segment='volume', **additional)


if __name__ == '__main__':

    fset = FrameSet()

    fmag = FerroMagnetic(*fset.frames)
    fmag.insert(0, 0, 5, 0.2)
    fmag.insert(2, 1, 2, volume=0.1)
    fmag.insert(1, 3, 2, volume=5)

    coil = Coil(*fset.frames, 0.5)
    coil.insert(1, 5, dl=0.3, dt=0.1)

    fset.frame.vtkplot()
