"""Biot specific Frame class."""
import numpy as np
import numpy.typing as npt

from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.framelink import FrameLink
from nova.electromagnetic.biotsection import BiotSection
from nova.electromagnetic.biotshape import BiotShape
from nova.electromagnetic.biotreduce import BiotReduce
from nova.electromagnetic.metaframe import MetaFrame


# pylint: disable=too-many-ancestors


class BiotFrame(FrameSpace):
    """Extend FrameSpace class with biot specific attributes and methods."""

    _metadata = ['turns', 'reduce']

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(BiotShape, BiotSection, BiotReduce)
        if isinstance(data, FrameLink):
            self.attrs['frame'] = data

    def extract_attrs(self, data, attrs):
        """Extend FrameAttrs.extract_attrs, lanuch custom metaframe."""
        if not self.hasattrs('metaframe'):
            self.attrs['metaframe'] = MetaFrame(
                self.index, required=['x', 'z'],
                additional=['plasma', 'nturn', 'link', 'segment'],
                available=['section', 'poly'],
                subspace=['segment'])
        super().extract_attrs(data, attrs)

    def __call__(self, attr) -> npt.ArrayLike:
        """Return flattened attribute matrix, shape(source*target,)."""
        vector = np.array(getattr(self, attr))
        region = self.biotshape.region
        if self.biotshape.region == '':
            raise IndexError('Frame region not specified.\n'
                             'Define partner source or target number.\n'
                             'self.set_target(number)\n'
                             'self.set_source(number)')
        assert region in ['source', 'target']
        if region == 'source':
            return np.dot(np.ones((self.biotshape.target, 1)),
                          vector.reshape(1, -1)).flatten()
        return np.dot(vector.reshape(-1, 1),
                      np.ones((1, self.biotshape.source))).flatten()

    def insert(self, *required, iloc=None, **additional):
        """Extend FrameLink.insert. Store referance to parent frame."""
        if len(required) > 0:
            if isinstance(required[0], FrameLink) and len(self) == 0:
                self.attrs['frame'] = required[0]
            if self.hasattrs('frame') and len(self) > 0:
                del self.attrs['frame']
        return super().insert(*required, iloc=iloc, **additional)

    def set_target(self, number):
        """Set target number."""
        return self.biotshape.set_target(number)

    def set_source(self, number):
        """Set source number."""
        return self.biotshape.set_source(number)


if __name__ == '__main__':

    biotframe = BiotFrame()
    biotframe.insert(range(3), 0, dl=0.95, dt=0.95, section='hex')
    biotframe.insert(range(3), 1, dl=0.95, dt=0.95, section='circ', link=True)
    biotframe.insert(range(3), 2, dl=0.95, dt=0.95, section='sq', link=False)
    biotframe.insert(range(3), 3, dl=0.95, dt=0.6, section='sk', link=True)

    biotframe.multipoint.link(['Coil0', 'Coil11', 'Coil2', 'Coil8'])

    biotframe.polyplot()



'''

    def update(self):
        """Update self."""
        self.drop()
        self.insert(self.frame)
        if self.frameindex != slice(None):
            self.index_coil(self.frameindex)

    def index_coil(self, index):
        """
        Drop coils, index coilframe.

        Parameters
        ----------
        index : Union[int, str, list[int], list[bool], list[str],
                      pandas.Index, slice]
            Coil index.

        Returns
        -------
        None.

        """
        if not isinstance(index, pandas.Index):
            index = self.coilframe.index[index]
        index = np.array([name in index for name in self.coilframe.index])
        self._frameindex = index[self.coilframe._reduction_index]
        drop_index = self.index[~index]
        CoilFrame.drop_coil(self, drop_index)
        self._update_cross_section_factor()

    def _link_coilframe(self, *args):
        """Link to coilframe instance to propagate future coilframe updates."""
        if self._is_coilframe(*args, accept_dataframe=False):
            self.coilframe = args[0]

    def update_coilframe(self):
        """
        Rebuild coilframe following geometric changes to coilset.

        Returns
        -------
        None.

        """
        if hasattr(self, 'coilframe'):
            if self.coilframe is not None:
                if self.coilframe.nC != self._framenumber:
                    self.update_coil()
'''
