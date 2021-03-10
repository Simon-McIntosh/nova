import pytest


from nova.electromagnetic.frame import Frame


def test_fix_aspect():
    frame = Frame()
    frame.add_frame(4, 6, 0.1, 0.5, section='sq')
    assert frame.dx[0] == frame.dz[0]


def test_free_aspect():
    frame = Frame()
    frame.add_frame(4, 6, 0.1, 0.5, section='r')
    assert frame.dx[0] != frame.dz[0]


if __name__ == '__main__':

    pytest.main([__file__])
