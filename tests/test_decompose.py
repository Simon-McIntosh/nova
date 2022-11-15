import pytest

import numpy as np

from nova.linalg.decompose import Decompose


rng = np.random.default_rng(seed=2025)
matricies = [rng.random((200, 110)), rng.random((70, 130)),
             rng.random((100, 100))]


@pytest.mark.parametrize('matrix', matricies)
def test_default_rank(matrix):
    decompose = Decompose(matrix)
    assert np.allclose(decompose.matrix,
                       decompose['U'] * decompose['s'] @ decompose['Vh'])


if __name__ == '__main__':

    pytest.main([__file__])
