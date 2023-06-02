import numpy as np
import pytest

from nova.geometry.strike import Strike
from nova.imas.utilities import ids_attrs, mark


@mark["wall"]
def test_divertor_multi():
    strike = Strike(wall=ids_attrs["wall"])
    strike.update([np.array([(3.5, -3), (7, -5)])])
    assert len(strike.points) == 4


@mark["wall"]
def test_divertor_single():
    strike = Strike(wall=ids_attrs["wall"])
    strike.update([np.array([(3.5, -3), (7, -3)])])
    assert len(strike.points) == 1


@mark["wall"]
def test_main_chamber_number():
    strike = Strike(wall=ids_attrs["wall"], indices=(0,))
    strike.update([np.array([(3.5, -3), (7, -5)])])
    assert len(strike.points) == 0


@mark["wall"]
def test_main_multi_limiter():
    strike = Strike(wall=ids_attrs["wall"], indices=(0, 1))
    strike.update([np.array([(5, -4), (5, 5)])])
    assert len(strike.points) == 2


if __name__ == "__main__":
    pytest.main([__file__])
