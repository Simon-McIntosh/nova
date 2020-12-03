
import os.path
import pytest
import numpy as np

from nova.thermalhydralic.localdata import LocalData


def test_localdata_experiment():
    local = LocalData('CSJA_6')
    local.experiment = 'CSJA_3'
    assert local.experiment == 'CSJA_3'


def test_make_directory():
    local = LocalData('CSJA_6')
    local.makedir()
    isdir = os.path.isdir(local.experiment_directory)
    local.removedir()
    isnotdir = ~os.path.isdir(local.experiment_directory)
    assert np.array([isdir, isnotdir]).all()


if __name__ == '__main__':
    pytest.main([__file__])
