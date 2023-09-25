import pytest

import numpy as np

from nova.biot.arc import Arc
from nova.frame.coilset import CoilSet


@pytest.mark.skip("pending refactor of BiotFrame methods for 3D elements")
def test_transform():
    coilset = CoilSet(dwinding=0, field_attrs=["Bx"])
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 5, 3.2], [-5, 0, 3.2]])
    )
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 0, -1.8], [-5, 0, 3.2]])
    )
    arc = Arc(coilset.subframe)
    assert np.allclose(arc.start_point, arc._to_global(arc._to_local(arc.start_point)))
    assert np.isclose(
        arc._to_local(arc.start_point)[0, 2], arc._to_local(arc.end_point)[0, 2]
    )
    assert np.shape(arc.transform) == (2, 3, 3)
    assert np.allclose(arc._to_local(arc.axis)[1], [0, 0, 1])


if __name__ == "__main__":
    pytest.main([__file__])
