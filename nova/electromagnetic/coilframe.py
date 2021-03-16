"""Extend pandas.DataFrame to manage coil and subcoil data."""

from typing import Optional, Collection, Any, Union

import numpy as np

from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.frame import Frame
from nova.electromagnetic.subspace import SubSpace

# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object


class CoilFrame(Frame):
    """
    Extend SuperSpace.

    - Implement current properties.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Union[MetaArray, MetaFrame]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.attrs['subspace'] = SubSpace(self)


if __name__ == '__main__':

    coilframe = CoilFrame(Required=['x', 'z'], optimize=True,
                          dCoil=5, Additional=['Ic'], Nt=10)

    coilframe.add_frame(4, range(3), link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.add_frame(4, range(4), link=True)

    def set_current():
        coilframe.Ic = np.random.rand(len(coilframe.subspace))

    set_current()

    coilframe.subspace.iloc[3, 0] = 33.4
    print(coilframe)

    #frame.x = [1, 2, 3]
    #frame.x[1] = 6

    #print(frame)

    #frame.metaarray._lock = False
    #newframe = Frame()

