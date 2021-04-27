import pytest

from nova.electromagnetic.frameset import FrameSet 
from nova.electromagnetic.error import SpaceKeyError


def test_space_setattr_error():
    frameset = FrameSet(required=['rms'], additional=['Ic'])
    frameset.subframe.insert([2, 4], It=6, link=True)
    with pytest.raises(SpaceKeyError):
        frameset.subframe.Ic = 7

    
if __name__ == '__main__':
    pytest.main([__file__])
    