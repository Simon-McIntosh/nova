"""Extend DataFrame - add subspace."""

from typing import Optional, Collection, Any

import numpy as np

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metamethod import MetaMethod

# pylint: disable=too-many-ancestors


class Frame(FrameSet):
    """
    Extend DataFrame.

    - Implement subspace.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, MetaMethod] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.attrs['subspace'] = SubSpace(self)

    def drop(self, index=None):
        """Drop frame(s), re-initialize subspace."""
        super().drop(index)
        self.subspace = SubSpace(self)


def set_current():
    """Test current update with randomized input (check update speed)."""
    # frame.subspace.metaarray.data['Ic'] = np.random.rand(len(frame.subspace))
    frame.Ic = np.random.rand(len(frame.subspace))
    # _ = frame.Ic


if __name__ == '__main__':

    frame = Frame(Required=['x', 'z'], Additional=[], Array=['Ic'])
    frame.insert([4, 5], 1, Ic=6.5)
    frame.insert(1, range(4000), It=5, link=True)
    print(frame)
