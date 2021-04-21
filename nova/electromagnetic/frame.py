"""Extend DataFrame - add subspace."""

from typing import Optional, Collection, Any

import numpy as np

from nova.electromagnetic.dataframe import SubSpaceLockError
from nova.electromagnetic.framearray import (
    FrameArray, FrameArrayLocMixin, FrameArrayIndexer
    )
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.select import Select
from nova.electromagnetic.geometry import Geometry
from nova.electromagnetic.polyplot import PolyPlot


# pylint: disable=too-many-ancestors


class FrameLocMixin(FrameArrayLocMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __getitem__(self, key):
        """Inflate subspace items prior to return."""
        col = self.obj.get_col(key)
        if self.obj.metaframe.hascol('subspace', col):
            if self.obj.lock('subspace') is False:
                self.obj.inflate_subspace(col)
        elif col == 'It' and self.obj.metaframe.hascol('subspace', 'Ic'):
            if self.obj.lock('subspace') is False:
                self.obj.inflate_subspace('Ic')
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj.get_col(key)
        value = self.obj.format_value(col, value)
        if self.obj.metaframe.hascol('subspace', col):
            if self.obj.lock('subspace') is False:
                raise SubSpaceLockError(self.name, col)
        return super().__setitem__(key, value)


class FrameIndexer(FrameArrayIndexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return FrameLocMixin


class Frame(FrameIndexer, FrameArray):
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
        """Extend FrameArray add_methods, add additional methods to attrs."""
        self.attrs['select'] = Select(self)
        self.attrs['geom'] = Geometry(self)
        self.attrs['polyplot'] = PolyPlot(self)
        super().add_methods()

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if not self.hasattrs('subspace'):
            return super().__getitem__(col)
        if self.metaframe.hascol('subspace', col):
            if self.lock('subspace') is False:
                self.inflate_subspace(col)
        elif col == 'It' and self.metaframe.hascol('subspace', 'Ic'):
            if self.lock('subspace') is False:
                self.inflate_subspace('Ic')
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self.format_value(col, value)
        if not self.hasattrs('subspace'):
            return super().__setitem__(col, value)
        if self.metaframe.hascol('subspace', col):
            if self.lock('subspace') is False:
                raise SubSpaceLockError('setitem', col)
        return super().__setitem__(col, value)

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if self.hasattrs('subspace'):
            for col in [col for col in self.subspace if col in self]:
                self.inflate_subspace(col)
        super().update_frame()  # update dataarray

    def inflate_subspace(self, col):
        """Inflate subspace variable and setattr in frame."""
        with self.setlock(False, 'subspace'):
            value = self.subspace.__getitem__(col)
        if not isinstance(value, np.ndarray):
            value = value.to_numpy()
        try:
            value = value[self.subref]  # inflate if subref set
            if col == 'Ic':
                value *= self.factor.values
        except (AttributeError, IndexError):
            pass
        with self.setlock(True, 'subspace'):
            super().__setitem__(col, value)


def set_current():
    """Test current update with randomized input (check update speed)."""
    # frame.subspace.metaframe.data['Ic'] = np.random.rand(len(frame.subspace))
    frame.subspace.Ic = np.random.rand(len(frame.subspace))
    # _ = frame.Ic

def get_current():
    _ = frame.loc[:, 'Ic']
    #_ = frame['Ic']


if __name__ == '__main__':

    frame = Frame(required=['x', 'z'],
                  Available=['It'],
                  Subspace=[],
                  Array=['Ic'])
    frame.insert([-4, -5], 1, Ic=6.5, name='PF1', active=False, plasma=True)# label='CS')
    #frame.insert(range(4000), 3, Ic=4, nturn=20, label='PF', link=True)
    #frame.multipoint.link(['PF1', 'CS0'], factor=1)
    #print(frame.loc[:, ['active', 'passive', 'plasma', 'coil']])

    for _ in range(1000):
        get_current()
