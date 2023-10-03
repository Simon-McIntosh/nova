import pytest

import numpy as np

from nova.biot.biotframe import Source, Target


def test_transform():
    source = Source()
    target = Target({"x": np.arange(5, 7.5, 10), "z": 0.5}, Available=[])
    print(target)


if __name__ == "__main__":
    pytest.main([__file__])
