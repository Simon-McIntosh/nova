"""Manage frame super/sub space. Protect access to subspace attrs via frame."""

from typing import Optional, Collection, Any, Union

import pandas
import numpy as np

from nova.electromagnetic.superframe import SuperFrame
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame

# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object
# pylint: disable=protected-access


class SuperSpace(SuperFrame):
    """
    Extend SuperFrame to implement super/sub space access.

    - Protect setitem operations on loc, iloc, at, and iat operators.
    - Extend set/getattr and set/getitem to serve subspace.
    - Extend repr to update superframe prior to printing.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Union[MetaArray, MetaFrame]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.attrs['subspace'] = SubSpace(self)

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def in_subspace(self, col):
        """Return Ture if col in metaframe.subspace."""
        if isinstance(col, int):
            col = self.columns[col]
        if not isinstance(col, str):
            return False
        if 'metaframe' in self.attrs and 'subspace' in self.attrs:
            return col in self.metaframe.subspace
        return False

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if hasattr(self, 'subspace'):
            for col in self.subspace:
                self.set_frame(col)

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.check_subspace(col)
        with self.metaframe.setlock('subspace', True):
            value = getattr(self, col).to_numpy()[self.subref]
        with self.metaframe.setlock('subspace', None):
            super().__setattr__(col, value)

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.check_subspace(col)
        with self.metaframe.setlock('subspace', False):
            return super().__getattr__(col)

    def check_subspace(self, col):
        """Check for col in metaframe.subspace, raise error if not found."""
        if not self.in_subspace(col):
            raise IndexError(f'\'{col}\' not specified in metaframe.subspace '
                             f'{self.metaframe.subspace}')

    def __getattr__(self, col):
        """Extend DataFrame.__getattr__. (frame.*)."""
        if col in self.attrs:
            return self.attrs[col]
        if self.in_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        return super().__getattr__(col)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.in_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        return super().__getitem__(col)

    def __setattr__(self, col, value):
        """Check lock. Extend DataFrame.__setattr__ (frame.* = *).."""
        value = self._format_value(col, value)
        if self.in_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setattr__(col, value)
            if self.metaframe.lock('subspace') is False:
                raise SuperSpaceIndexError('setattr', col)
        return super().__setattr__(col, value)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self._format_value(col, value)
        if self.in_subspace(col):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setitem__(col, value)
            if self.metaframe.lock('subspace') is False:
                raise SuperSpaceIndexError('setitem', col)
        return super().__setitem__(col, value)

    def _format_value(self, col, value):
        if not pandas.api.types.is_numeric_dtype(type(value)) \
                or 'metaframe' not in self.attrs:
            return value
        try:
            dtype = type(self.metaframe.default[col])
        except KeyError:  # no default type
            return value
        try:
            if pandas.api.types.is_list_like(value):
                return np.array(value, dtype)
            return dtype(value)
        except (ValueError, TypeError):  # NaN conversion error
            return value


if __name__ == '__main__':

    superspace = SuperSpace(Required=['x', 'z'], Additional=['Ic'])

    superspace.add_frame(4, range(3), link=True)
    '''
    superspace.add_frame(4, range(2), link=False)
    superspace.add_frame(4, range(40), link=True)

    superspace.subspace.loc[:, 'Ic'] = 12
    superspace.subspace.loc['Coil5', 'Ic'] = 5.4

    print(superspace)
    '''
