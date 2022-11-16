import pytest
import tempfile

import numpy as np

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.error import SpaceKeyError


def test_space_setattr_error():
    frameset = FrameSet(required=['rms'], additional=['Ic'])
    frameset.subframe.insert([2, 4], It=6, link=True)
    with pytest.raises(SpaceKeyError):
        frameset.subframe.Ic = 7


def test_store_load():
    frameset = FrameSet(required=['rms'], additional=['Ic'])
    frameset.subframe.insert([2, 4], It=6, link=True)

    subframe = frameset.subframe
    with tempfile.NamedTemporaryFile() as tmp:
        frameset.store(tmp.name)
        del frameset
        frameset = FrameSet().load(tmp.name)
    assert (frameset.subframe.link == subframe.link).all()
    assert np.isclose(frameset.sloc['Ic'], [6]).all()


def test_subspace_dataframe_access():
    frameset = FrameSet(required=['x', 'z'], additional=['Ic'],
                        subspace=['Ic'])
    frameset.subframe.insert(2, range(2), Ic=0)
    frameset.sloc['Ic'] = 10
    assert frameset.sloc[:, ['Ic']].squeeze().to_list() == [10, 10]


if __name__ == '__main__':

    pytest.main([__file__])
