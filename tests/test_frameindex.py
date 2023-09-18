import pytest

import numpy as np

from nova.frame.framespace import FrameSpace


def test_get_loc():
    framespace = FrameSpace(
        required=["x"], additional=["dy"], array=["dy"], label="Coil"
    )
    framespace.insert(5 * np.ones(7), dy=range(7))
    assert np.allclose(framespace.loc[:, "dy"], range(7))
    assert np.allclose(framespace.loc["Coil0":"Coil6", "dy"], range(7))
    assert np.allclose(framespace.loc["Coil3":"Coil5", "dy"], [3, 4, 5])
    assert np.allclose(framespace.loc[["Coil3", "Coil5"], "dy"], [3, 5])


if __name__ == "__main__":
    pytest.main([__file__])
