"""Extend DataFrame - add subspace."""

from typing import Optional, Collection, Any

import numpy as np

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metamethod import MetaMethod

# pylint: disable=too-many-ancestors


class Frame(DataFrame):
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

    frame = Frame(Required=['x', 'z'], Additional=['It'])
    frame.add_frame(1, 3, It=5)
    frame.loc[:, 'x'] = 7
    print(frame)