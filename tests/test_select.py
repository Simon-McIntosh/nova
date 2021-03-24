import pytest

from nova.electromagnetic.frameset import FrameSet


def test_additional():
    frame = FrameSet(Required=['x'])
    frame.insert(range(3), active=True, link=True)
    frame.insert(range(3), active=False, link=True)
    frame.insert(range(2), plasma=True, link=True)
    print(frame.select('plasma'))


if __name__ == '__main__':

    test_additional()