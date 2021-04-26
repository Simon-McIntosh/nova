"""Biot specific Frame class."""
import numpy as np

from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.framelink import FrameLink
from nova.electromagnetic.biotsection import BiotSection
from nova.electromagnetic.biotshape import BiotShape


# pylint: disable=too-many-ancestors


class BiotFrame(FrameSpace):
    """Extend CoilFrame class with biot specific attributes and methods."""

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        metadata = {'required': ['x', 'z'],
                    'available': ['link', 'section', 'poly']} | metadata
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(BiotShape, BiotSection)
        if isinstance(data, FrameLink):
            self.attrs['frame'] = data

    def __call__(self, attr):
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

    frame = FrameSpace(required=['x'])
    frame.insert(range(3), dl=0.95, dt=0.95, section='hex',
                 turn='hex')
    biotframe = BiotFrame(frame)
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