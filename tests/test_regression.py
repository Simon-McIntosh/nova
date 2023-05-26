import pytest

import numpy as np

from nova.utilities.importmanager import mark_import
with mark_import('optimize') as mark_optimize:
    from nova.linalg.lops import Lops
    from pylops.utils import dottest

from nova.linalg.decompose import Decompose
from nova.linalg.regression import OdinaryLeastSquares, MoorePenrose


matrix_shapes = [(3, 3), (2, 9), (5, 12)]
rng_seed = 2025


def test_svd_matrices():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    svd = Decompose(matrix, rank=0)
    assert all([attr in svd.matrices for attr in ['U', 's', 'Vh', 'Uh', 'V']])


def test_model_init():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    model = rng.random(matrix.shape[1])
    ols = OdinaryLeastSquares(matrix, model=model)
    assert id(model) == id(ols.model)


def test_model_update():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    model = rng.random(matrix.shape[1])
    ols = OdinaryLeastSquares(matrix)
    ols.forward(model)
    assert id(model) == id(ols.model)


def test_coordinate_update_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[1])
    coordinate = np.linspace(0, 1, matrix_shapes[0][0])
    with pytest.raises(IndexError):
        OdinaryLeastSquares(matrix, coordinate)


def test_model_init_index_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[1])
    model = rng.random(matrix.shape[0])
    with pytest.raises(IndexError):
        OdinaryLeastSquares(matrix, model=model)


def test_model_update_index_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[1])
    model = rng.random(matrix.shape[0])
    ols = OdinaryLeastSquares(matrix)
    with pytest.raises(IndexError):
        ols.update_model(model)


def test_data_init():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    data = rng.random(matrix.shape[0])
    ols = OdinaryLeastSquares(matrix, data=data)
    assert id(data) == id(ols.data)


def test_data_update_index_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[1])
    data = rng.random(matrix.shape[1])
    with pytest.raises(IndexError):
        OdinaryLeastSquares(matrix, data=data)


def test_no_model_attribute_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    ols = OdinaryLeastSquares(matrix)
    with pytest.raises(AttributeError):
        ols.update_model(None)


def test_no_data_attribute_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    ols = OdinaryLeastSquares(matrix)
    with pytest.raises(AttributeError):
        ols.update_data(None)


def test_forward_no_model_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    ols = OdinaryLeastSquares(matrix)
    with pytest.raises(AttributeError):
        ols.forward()


def test_forward_no_data_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    ols = OdinaryLeastSquares(matrix)
    with pytest.raises(AttributeError):
        ols.inverse()


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_forward(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    model = rng.random(matrix.shape[1])
    ols = OdinaryLeastSquares(matrix, model=model)
    assert np.allclose(matrix @ model, ols.forward())


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_inverse(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    data = rng.random(matrix.shape[0])
    ols = OdinaryLeastSquares(matrix, data=data)
    assert np.allclose(np.linalg.lstsq(matrix, data, rcond=None)[0],
                       ols.inverse())


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_regression_dot(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    model = rng.random(matrix.shape[1])
    data = rng.random(matrix.shape[0])
    ols = OdinaryLeastSquares(matrix, model=model, data=data)

    forward = ols.forward()
    adjoint = ols.adjoint()

    forward_data = forward @ data
    model_adjoint = model @ adjoint
    assert (forward_data - model_adjoint) / (
        (forward_data + model_adjoint + 1e-15) / 2) < 1e-15


@mark_optimize
@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_regression_lops_dot(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    lops = Lops(matrix)
    dottest(lops, *matrix_shape)


@mark_optimize
@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_regression_lops(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    data = rng.random(matrix.shape[0])
    ols = OdinaryLeastSquares(matrix, data=data)
    lops = Lops(matrix)
    assert np.allclose(ols.inverse(), lops / data)


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_moore_penrose(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    data = rng.random(matrix.shape[0])
    ols = OdinaryLeastSquares(matrix)
    ols /= data
    mpen = MoorePenrose(matrix, rank=min(matrix_shape))
    mpen /= data
    assert np.allclose(ols.model, mpen.model)


if __name__ == '__main__':

    pytest.main([__file__])
