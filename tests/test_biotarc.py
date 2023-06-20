from itertools import product
import pytest

import numpy as np
from scipy.special import ellipkinc, ellipj


@pytest.mark.parametrize(
    "k,theta", product([0.05, 0.3, 0.8], [0, np.pi / 3, np.pi / 2])
)
def test_jacobi_amplitude(k, theta):
    u = ellipkinc(theta, k**2)  # Jacobi amplitude
    phi = ellipj(u, k**2)[-1]
    assert np.isclose(phi, theta)


if __name__ == "__main__":
    pytest.main([__file__])
