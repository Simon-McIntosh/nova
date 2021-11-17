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


if __name__ == '__main__':

    pytest.main([__file__])
