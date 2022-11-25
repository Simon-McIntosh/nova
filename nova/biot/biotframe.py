"""Biot specific Frame class."""
import dask.array as da
import numpy as np

from nova.frame.framespace import FrameSpace
from nova.biot.biotsection import BiotSection
from nova.biot.biotshape import BiotShape
from nova.biot.biotreduce import BiotReduce
from nova.frame.geometry import PolyGeo


# pylint: disable=too-many-ancestors


class BiotFrame(FrameSpace):
    """Extend FrameSpace class with biot specific attributes and methods."""

    _metadata = ['turns', 'reduce']

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(PolyGeo, BiotShape, BiotSection, BiotReduce)
        for attr in self._metadata:
            setattr(self, attr, None)

    def update_metadata(self, data, columns, attrs, metadata):
        """Extend FrameAttrs update_metadata."""
        metadata = dict(required=['x', 'z'],
                        additional=['plasma', 'nturn', 'link', 'segment'],
                        available=['section', 'poly'],
                        array=['x', 'z', 'dx', 'dz',
                               'area', 'nturn']) | metadata
        super().update_metadata(data, columns, attrs, metadata)

    def __call__(self, attr) -> da.Array | np.ndarray:
        """Return attribute matrix, shape(target, source)."""
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
        matrix = np.tile(self[attr], reps=(reps, 1))
        if region == 'target':
            matrix = np.transpose(matrix)
        return matrix

    def set_target(self, number):
        """Set target number."""
        return self.biotshape.set_target(number)

    def set_source(self, number):
        """Set source number."""
        return self.biotshape.set_source(number)


class BiotTarget(BiotFrame):
    """Extend BiotFrame dropping additional and available metadata."""

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        for attr in ['additional', 'available']:
            metadata[attr] = []
        super().__init__(data, index, columns, attrs, **metadata)


if __name__ == '__main__':

    biotframe = BiotFrame()
    biotframe.insert(range(3), 0, dl=0.95, dt=0.95, section='hex')
    biotframe.insert(range(3), 1, dl=0.95, dt=0.95, section='disc', link=True)
    biotframe.insert(range(3), 2, dl=0.95, dt=0.95, section='square',
                     link=False)
    biotframe.insert(range(3), 3, dl=0.95, dt=0.6, section='skin', link=True)

    biotframe.multipoint.link(['Coil0', 'Coil11', 'Coil2', 'Coil8'])
