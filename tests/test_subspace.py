
import pytest

from nova.electromagnetic.frame import Frame


def test_init():
    frame = Frame(link=True, metadata={'additional': ['link']})
    frame.add_frame(4, [5, 7, 12], 0.1, 0.05)
    return frame


def test_range_len():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(30), link=True)
    frame.add_frame(4, range(2), link=False)
    frame.add_frame(4, range(4), link=True)
    assert len(frame.range) == 4 and len(frame) == 36


if __name__ == '__main__':

    pytest.main([__file__])
