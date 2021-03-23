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


if __name__ == '__main__':


    frame = Frame(Required=['x', 'z'], Additional=['section', 'link', 'Ic'])
    frame.add_frame(1, range(3), It=5, link=True)
    print(frame)
    #frame.add_frame(1, range(4000), Ic=5.7, Nt=334.5, link=True)
    #frame.loc[:, 'Ic'] = 7

    #frame.Ic = np.random.rand(len(frame.subspace))
    #frame.Ic = 9
    #print(frame.columns)

    '''
    def set_current():
        #frame.subspace.metaarray.data['Ic'] = np.random.rand(len(frame.subspace))
        frame.Ic = np.random.rand(len(frame.subspace))
        #_ = frame.Ic

    for _ in range(4000):
        set_current()

    print(frame)

    '''



