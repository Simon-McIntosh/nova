
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas
import shapely.geometry
import shapely.strtree
import scipy.interpolate

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.frameattrs import FrameAttrs
from nova.electromagnetic.polygen import PolyFrame
from nova.utilities import geom


@dataclass
class Shell(FrameAttrs):
    """Mesh poloidal shell elements."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float
    turn: str = 'shell'
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Shl'})

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for shell."""

    def insert(self, *required, iloc=None, **additional):
        """
        Add shell elements to frameset.

        Lines described by x, z coordinates meshed into n coils based on
        dshell. Each frame is meshed based on delta.

        Parameters
        ----------
        x : array-like, shape(n,)
            x-coordinates of poloidal line to be meshed.
        z : array-like, shape(n,)
            z-coordinates of poloidal line to be meshed.
        dt : float
            Shell thickness.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """

        shell, subshell = self._mesh((x, z), dt, rho, attrs.pop('dshell'))
        additional = shell | attrs
        required = [additional.pop(attr)
                    for attr in self.frame.metaframe.required]

        index = self.frame.insert(*required, **additional)

        attrs |= {'delim': '_'}
        subframe = []
        for i, frame in enumerate(index):
            shell = self._mesh(
                subshell['segment'][i], subshell['dt'][i],
                subshell['rho'][i], attrs['delta'])[0]
            additional = shell | attrs
            additional |= {'frame': frame, 'label': frame, 'link': True}

            required = [additional.pop(attr)
                        for attr in self.frame.metaframe.required]
            subframe.append(self.subframe.assemble(*required, **additional))

        self.subframe.concatenate(*subframe)
