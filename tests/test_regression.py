import pytest

import numpy as np
import pylops

from nova.linalg.decompose import Decompose
from nova.linalg.regression import Regression, RegressionLops


matrix_shapes = [(3, 3), (2, 9), (5, 12)]
rng_seed = 2025


def test_svd_matrices():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    svd = Decompose(matrix, rank=None)
    assert all([attr in svd.matrices for attr in ['U', 's', 'Vh', 'Uh', 'V']])


def test_model_init():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    model = rng.random(matrix.shape[1])
    regression = Regression(matrix, model)
    assert id(model) == id(regression.model)


def test_model_update():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    model = rng.random(matrix.shape[1])
    regression = Regression(matrix)
    regression.forward(model)
    assert id(model) == id(regression.model)


def test_model_init_index_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[1])
    model = rng.random(matrix.shape[0])
    with pytest.raises(IndexError):
        Regression(matrix, model)


def test_model_update_index_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[1])
    model = rng.random(matrix.shape[0])
    regression = Regression(matrix)
    with pytest.raises(IndexError):
        regression.update_model(model)


def test_data_init():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    data = rng.random(matrix.shape[0])
    regression = Regression(matrix, data=data)
    assert id(data) == id(regression.data)


def test_data_update_index_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[1])
    data = rng.random(matrix.shape[1])
    with pytest.raises(IndexError):
        Regression(matrix, data=data)


def test_no_model_attribute_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    regression = Regression(matrix)
    with pytest.raises(AttributeError):
        regression.update_model(None)


def test_no_data_attribute_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    regression = Regression(matrix)
    with pytest.raises(AttributeError):
        regression.update_data(None)


def test_forward_no_model_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    regression = Regression(matrix)
    with pytest.raises(AttributeError):
        regression.forward()


def test_forward_no_data_error():
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shapes[0])
    regression = Regression(matrix)
    with pytest.raises(AttributeError):
        regression.inverse()


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_forward(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    model = rng.random(matrix.shape[1])
    regression = Regression(matrix, model)
    assert np.allclose(matrix @ model, regression.forward())


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_inverse(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    data = rng.random(matrix.shape[0])
    regression = Regression(matrix, data=data)
    assert np.allclose(np.linalg.lstsq(matrix, data)[0], regression.inverse())


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_regression_dot(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    model = rng.random(matrix.shape[1])
    data = rng.random(matrix.shape[0])
    regression = Regression(matrix, model, data)

    forward = regression.forward()
    adjoint = regression.adjoint()

    forward_data = forward @ data
    model_adjoint = model @ adjoint
    assert (forward_data - model_adjoint) / (
        (forward_data + model_adjoint + 1e-15) / 2) < 1e-15


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_regression_lops_dot(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    lops = RegressionLops(matrix)
    pylops.utils.dottest(lops, *matrix_shape)


@pytest.mark.parametrize('matrix_shape', matrix_shapes)
def test_regression_lops(matrix_shape):
    rng = np.random.default_rng(seed=rng_seed)
    matrix = rng.random(matrix_shape)
    data = rng.random(matrix.shape[0])
    regression = Regression(matrix, data=data)
    lops = RegressionLops(matrix)
    assert np.allclose(regression.inverse(), lops / data)


if __name__ == '__main__':

    pytest.main([__file__])
