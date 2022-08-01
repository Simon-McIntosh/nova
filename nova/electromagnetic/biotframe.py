"""Biot specific Frame class."""
import dask.array as da
import numpy as np

from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.biotsection import BiotSection
from nova.electromagnetic.biotshape import BiotShape
from nova.electromagnetic.biotreduce import BiotReduce
from nova.electromagnetic.geometry import PolyGeo


# pylint: disable=too-many-ancestors


class BiotFrame(FrameSpace):
    """Extend FrameSpace class with biot specific attributes and methods."""

    _metadata = ['turns', 'reduce']

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(PolyGeo, BiotShape, BiotSection, BiotReduce)

    def update_metadata(self, data, columns, attrs, metadata):
        """Extend FrameAttrs update_metadata."""
        metadata = dict(required=['x', 'z'],
                        additional=['plasma', 'nturn', 'link', 'segment'],
                        available=['section', 'poly'],
                        array=['x', 'z']) | metadata
        super().update_metadata(data, columns, attrs, metadata)

    def __call__(self, attr, chunks=1000) -> da.Array:
        """Return attribute matrix, shape(target, source)."""
        vector = da.from_array(self[attr][:, np.newaxis], chunks=chunks)
        region = self.biotshape.region
        if self.biotshape.region == '':
            raise IndexError('Frame region not specified.\n'
                             'Define partner source or target number.\n'
                             'self.set_target(number)\n'
                             'self.set_source(number)')
        assert region in ['source', 'target']
        partner = next(partner for partner in
                       ['source', 'target'] if partner != region)
        reps = getattr(self.biotshape, partner)
        matrix = vector.map_blocks(np.tile, reps=reps, chunks=(chunks, chunks))
        if region == 'source':
            return da.transpose(matrix).compute_chunk_sizes()
        return matrix.compute_chunk_sizes()

    def set_target(self, number):
        """Set target number."""
        return self.biotshape.set_target(number)

    def set_source(self, number):
        """Set source number."""
        return self.biotshape.set_source(number)


if __name__ == '__main__':

    biotframe = BiotFrame()
    biotframe.insert(range(3), 0, dl=0.95, dt=0.95, section='hex')
    biotframe.insert(range(3), 1, dl=0.95, dt=0.95, section='disc', link=True)
    biotframe.insert(range(3), 2, dl=0.95, dt=0.95, section='square',
                     link=False)
    biotframe.insert(range(3), 3, dl=0.95, dt=0.6, section='skin', link=True)

    biotframe.multipoint.link(['Coil0', 'Coil11', 'Coil2', 'Coil8'])


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
