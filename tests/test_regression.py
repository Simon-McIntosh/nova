import pytest

import numpy as np

from nova.linalg.regression import Regression, Bernstein


def test_regression_abc():
    with pytest.raises(TypeError):
        _ = Regression(50, 3, True)


#bernstein = Bernstein(5, 3, model=np.arange(4, dtype=float))

#  TODO implement dot test (svd)

if __name__ == '__main__':

    pytest.main([__file__])
