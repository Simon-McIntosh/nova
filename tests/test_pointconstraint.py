import numpy as np
import pytest

from nova.imas.pulsedesign import Constraint


def test_constraint_point_index():
    constraint = Constraint(np.ones((8, 2)))
    constraint.poloidal_flux = 4, range(4)
    constraint.radial_field = 0, [0, 2]
    constraint.vertical_field = 0, [1, 3]
    constraint.radial_field = 0, [3]

    assert np.allclose(constraint.point_index, range(8))
    assert np.allclose(constraint.index("null"), 3)
    assert np.allclose(constraint.index("radial"), [0, 2])
    assert np.allclose(constraint.index("vertical"), 1)


if __name__ == "__main__":
    pytest.main([__file__])
