"""Extend pandas.DataFrame to manage coil and subcoil data."""

from typing import Optional, Collection, Any, Union

import pandas
import numpy as np

from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.superspace import SuperSpace


# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object

class Frame(SuperSpace):
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

    '''
    @property
    def line_current(self):
        return self.loc[:, 'Ic']

    @line_current.setter
    def line_current(self, current):
        self.loc[:, 'Ic'] = current
    '''

    def _current_label(self, **kwargs):
        """Return current label, Ic or It."""
        current_label = None
        if 'Ic' in self.metaframe.required or 'Ic' in kwargs:
            current_label = 'Ic'
        elif 'It' in self.metaframe.required or 'It' in kwargs:
            current_label = 'It'
        return current_label

    @staticmethod
    def _propogate_current(current_label, data):
        """
        "Propogate current data, Ic->It or It->Ic.

        Parameters
        ----------
        current_label : str
            Current label, Ic or It.
        data : Union[pandas.DataFrame, dict]
            Current / turn data.

        Returns
        -------
        None.

        """
        if current_label == 'Ic':
            data['It'] = data['Ic'] * data['Nt']
        elif current_label == 'It':
            data['Ic'] = data['It'] / data['Nt']


if __name__ == '__main__':

    frame = Frame(Required=['x', 'z'], optimize=True,
                  dCoil=5, Additional=['Ic'])

    frame.add_frame(4, range(3), link=True)
    frame.add_frame(4, range(2), link=False)
    frame.add_frame(4, range(4000), link=True)

    print(frame)
    nIc = len(frame.subspace)
    Ic = np.random.rand(nIc)

    def set_current():
        frame.Ic = Ic

    #frame.x = [1, 2, 3]
    #frame.x[1] = 6

    #print(frame)

    #frame.metaarray._lock = False
    #newframe = Frame()

