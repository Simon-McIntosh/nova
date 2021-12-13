"""Extend DataFrame - add subspace."""
import numpy as np

from nova.electromagnetic.framelink import FrameLink, LinkLocMixin, LinkIndexer
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.error import SpaceKeyError
from nova.electromagnetic.select import Select
from nova.electromagnetic.geometry import PolyGeo, VtkGeo
from nova.electromagnetic.polyplot import PolyPlot
from nova.electromagnetic.vtkplot import VtkPlot


# pylint: disable=too-many-ancestors


class SpaceLocMixin(LinkLocMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __getitem__(self, key):
        """
        Extend pandas.indexer getitem.

        Inflate subspace items prior to return.

        """
        col = self.obj.get_col(key)
        if self.obj.hascol('subspace', col):
            if self.obj.lock('subspace') is False:
                self.obj.inflate_subspace(col)
        elif col == 'It' and self.obj.hascol('subspace', 'Ic'):
            if self.obj.lock('subspace') is False:
                self.obj.inflate_subspace('Ic')
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """
        Extend pandas.indexer setitem.

        Protect subspace variable when set directly from frame.

        """
        col = self.obj.get_col(key)
        if self.obj.hascol('subspace', col):
            if self.obj.lock('subspace') is False:
                raise SpaceKeyError(self.name, col)
        return super().__setitem__(key, value)


class SpaceIndexer(LinkIndexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return SpaceLocMixin


class FrameSpace(SpaceIndexer, FrameLink):
    """
    Extend DataArray.

    - Implement subspace, selection and geometry methods.

    """

    def __init__(self, data=None, index=None, columns=None,
                 attrs=None, **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(Select, PolyGeo, VtkGeo, PolyPlot, VtkPlot)
        self.attrs['subspace'] = SubSpace(self)

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def __setattr__(self, col, value):
        """Extend DataFrame.__setattr__ to gain fast access to array data."""
        if self.hasattrs('subspace'):
            if self.hascol('subspace', col):
                if self.lock('subspace') is False:
                    raise SpaceKeyError('loc', col)
        return super().__setattr__(col, value)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.hasattrs('subspace'):
            if self.hascol('subspace', col):
                if self.lock('subspace') is False:
                    self.inflate_subspace(col)
            elif col == 'It' and self.hascol('subspace', 'Ic'):
                if self.lock('subspace') is False:
                    self.inflate_subspace('Ic')
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        if self.hasattrs('subspace'):
            if self.hascol('subspace', col):
                if self.lock('subspace') is False:
                    raise SpaceKeyError('loc', col)
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
        except (AttributeError, IndexError, TypeError):
            pass
        with self.setlock(True, 'subspace'):
            super().__setitem__(col, value)


def set_current():
    """Test current update with randomized input (check update speed)."""
    # framespace.subspace.metaframe.data['Ic'] = \
    #    np.random.rand(len(frame.subspace))
    framespace.subspace.Ic = np.random.rand(len(framespace.subspace))


def get_current():
    """Test current access."""
    _ = framespace.loc[:, 'Ic']  # 531 µs nC=40
    # _ = framespace['Ic']  # 86.5 µs nC=40
    # _ = framespace.Ic  # 98.6 µs nC=40
    # _ = framespace.subspace.Ic  # 31.5 µs nC=40


if __name__ == '__main__':

    framespace = FrameSpace(base=['x', 'y', 'z'],
                            required=['x', 'z'],
                            available=['It', 'poly'],
                            Subspace=['Ic'],
                            Array=['Ic'])
    framespace.insert(range(40), 1, Ic=6.5, name='PF1', part='PF',
                      active=False)

    #for _ in range(1000):
    #    get_current()

    '''
    framespace.insert(range(40), 3, Ic=4, nturn=20, label='PF', link=True)
    framespace.multipoint.link(['PF1', 'CS0'], factor=1)
    print(framespace.loc[:, ['active', 'passive', 'plasma', 'coil']])

    framespace.to_hdf('tmp.h5', 'frame')

    #del framespace
    #framespace = FrameSpace()
    #framespace.read_hdf('tmp.h5', 'frame')
    print(framespace.metaframe.metadata)

    for _ in range(1000):
        get_current()
    '''
