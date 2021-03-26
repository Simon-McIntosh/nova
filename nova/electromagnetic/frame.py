"""Extend DataFrame - add subspace."""

from typing import Optional, Collection, Any

import numpy as np

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.select import Select
from nova.electromagnetic.geometry import Geometry
from nova.electromagnetic.polyplot import PolyPlot


# pylint: disable=too-many-ancestors


class Frame(FrameSet):
    """
    Extend DataFrame.

    - Implement subspace, selection and geometory methods.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, MetaMethod] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.attrs['subspace'] = SubSpace(self)

    def add_methods(self):
        """Extend FrameSet add_methods, add additional methods to attrs."""
        self.attrs['select'] = Select(self)
        self.attrs['geom'] = Geometry(self)
        self.attrs['polyplot'] = PolyPlot(self)
        super().add_methods()


def set_current():
    """Test current update with randomized input (check update speed)."""
    # frame.subspace.metaarray.data['Ic'] = np.random.rand(len(frame.subspace))
    frame.Ic = np.random.rand(len(frame.subspace))
    # _ = frame.Ic


if __name__ == '__main__':

    frame = Frame(Required=['x', 'z'], available=['section'], Array=['Ic'])
    frame.insert([-4, -5], 1, Ic=6.5)
    frame.insert(*[x.flatten() for x in np.meshgrid(range(5), range(6))],
                 dl=0.75, dt=0.3, label='PF', frame='PF', link=True,
                 section='skin')

    frame.polyplot()


