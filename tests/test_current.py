
import pytest

from nova.electromagnetic.superframe import SuperFrame


def test_in_current():

    superframe = SuperFrame(Required=['x', 'z'], Additional=['Ic'])
    assert superframe.current.in_current('Ic')


if __name__ == '__main__':

    pytest.main([__file__])
