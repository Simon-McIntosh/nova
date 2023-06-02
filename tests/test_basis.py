import pytest

import numpy as np

from nova.linalg.basis import LinearSample, Basis, Bernstein, Svd

rng_seed = 2025


def test_linearsample():
    sample = LinearSample(32)
    assert np.allclose(sample.coordinate, np.linspace(0, 1, 32))


def test_basis_isabstract():
    with pytest.raises(TypeError):
        Basis()


def test_bernstein_shape():
    bernstein = Bernstein(25, 7)
    assert bernstein.shape == (25, 7 + 1)


def test_bernstein_sum():
    bernstein = Bernstein(25, 7)
    assert np.allclose(np.sum(bernstein.matrix, axis=1), 1)


def test_svd_shape():
    svd = Svd(33, 3)
    rng = np.random.default_rng(seed=rng_seed)
    svd += rng.random((25, 7))
    assert svd.shape == (33, 3)


def test_svd_iadd_invariance():
    svd = Svd(33, 3)
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random((25, 7))
    svd += matrix
    _matrix = svd.matrix.copy()
    svd += matrix
    assert np.allclose(abs(svd.matrix), abs(_matrix))


if __name__ == "__main__":
    pytest.main([__file__])
