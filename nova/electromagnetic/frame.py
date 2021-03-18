"""Extend DataFrame - add subspace."""

from typing import Optional, Collection, Any

import numpy as np

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.dataarray import DataArray
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metamethod import MetaMethod

# pylint: disable=too-many-ancestors


class Frame(DataArray):
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

    frame = Frame(Required=['x', 'z'], Additional=['It', 'Ic', 'Nt'],
                  Array=['Ic'])
    frame.add_frame(1, range(3), It=5, link=True)
    frame.add_frame(1, range(3), It=5, link=False)
    frame.add_frame(1, range(4000), It=5, Nt=334.5, link=True)
    #frame.loc[:, 'Ic'] = 7

    def set_current():
        #frame.metaarray.data['Ic'] = np.random.rand(len(frame.subspace))
        frame.Ic = np.random.rand(len(frame.subspace))

    for _ in range(4000):
        set_current()
    print(frame)
