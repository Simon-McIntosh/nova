"""Extend DataFrame - add subspace."""

from typing import Optional, Collection, Any

import numpy as np

from nova.electromagnetic.framearray import (
    FrameArray, FrameArrayLocMixin, FrameArrayIndexer
    )
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.select import Select
from nova.electromagnetic.geometry import Geometry
from nova.electromagnetic.polyplot import PolyPlot


# pylint: disable=too-many-ancestors

class SubSpaceError(IndexError):
    """Prevent direct access to frame's subspace variables."""

    def __init__(self, name, col):
        super().__init__(
            f'{name} access is restricted for subspace attributes. '
            f'Use frame.subspace.{name}[:, {col}] = *.\n\n'
            'Lock may be overridden via the following context manager '
            'but subspace will still overwrite (Cavieat Usor):\n'
            'with frame.setlock(None):\n'
            f'    frame.{name}[:, {col}] = *')


class FrameLocMixin(FrameArrayLocMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj.get_col(key)
        value = self.obj.format_value(col, value)
        if self.obj.metaframe.hascol('subspace', col):
            if self.obj.lock('subspace') is False:
                raise SubSpaceError(self.name, col)
        if isinstance(self.obj, SubSpace):
            print(col, self.obj.metaframe.subspace)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Inflate subspace items prior to return."""
        col = self.obj.get_col(key)
        if self.obj.metaframe.hascol('subspace', col):
            #if self.obj.lock('subspace') is False:
            #    key = self.obj.get_subkey(key)
            #    return getattr(self.obj.subspace, self.name)[key]
            #if self.obj.lock('subspace') is True:
            #self.obj.inflate_subspace(col)

            if self.obj.lock('subspace') is False:
                self.obj.inflate_subspace(col)
        elif col == 'It' and self.obj.metaframe.hascol('subspace', 'Ic'):
            if self.obj.lock('subspace') is False:
                self.obj.inflate_subspace('Ic')

        return super().__getitem__(key)


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

    def __setattr__(self, col, value):
        """Extend DataFrame.__setattr__ to provide access to subspace."""
        if self.metaframe.hascol('subspace', col):
            return self.subspace.__setattr__(col, value)
        return super().__setattr__(col, value)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.metaframe.hascol('subspace', col):
            if self.lock('subspace') is False:
                return self.subspace.__getitem__(col)
            if self.lock('subspace') is True:
                self.inflate_subspace(col)
        elif col == 'It' and self.metaframe.hascol('subspace', 'Ic'):
            if self.lock('subspace') is False:
                self.inflate_subspace('Ic')
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self.format_value(col, value)
        print(col, value)
        if self.hasattr('subspace'):
            if self.metaframe.hascol('subspace', col):
                if self.lock('subspace') is False:
                    return self.subspace.__setitem__(col, value)
                if self.lock('subspace') is True:
                    raise SubSpaceError('setitem', col)
        return super().__setitem__(col, value)

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if self.hasattr('subspace'):
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
        except AttributeError:
            pass
        with self.setlock(True, 'subspace'):
            super().__setitem__(col, value)


def set_current():
    """Test current update with randomized input (check update speed)."""
    # frame.subspace.metaarray.data['Ic'] = np.random.rand(len(frame.subspace))
    frame.Ic = np.random.rand(len(frame.subspace))
    # _ = frame.Ic


if __name__ == '__main__':

    frame = Frame(required=['x', 'z'], subspace=['Ic', 'It', 'Nt', 'z'])
    frame.insert([-4, -5], 1, Ic=6.5, label='CS')
    frame.insert([-4, -5], 3, Ic=3, label='PF', link=True)
    frame.multipoint.link([ 'PF1', 'CS0'], factor=1)

    print(frame)
    print()
    print(frame.subspace)




