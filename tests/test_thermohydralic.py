
import pytest

from nova.thermalhydralic.attributes import Attributes
from nova.thermalhydralic.localdata import LocalData


def test_set_attribute():
    da = Attributes()
    da.attributes = 'experiment'
    assert da.attributes[0] == 'experiment'


def test_set_attribute_array():
    da = Attributes()
    da.attributes = 'experiment'
    da.attributes = ['shot', 'testname']
    assert da.attributes == ['experiment', 'shot', 'testname']


def test_set_default_attribute():
    da = Attributes()
    da.default_attributes = {'read_txt': True}
    da.initialize_attributes()
    assert da._read_txt


def test_set_experiment():
    local = LocalData('CSJA_6')
    local.experiment = 'CSJA_3'
    assert local.experiment == 'CSJA_3'


if __name__ == '__main__':
    pytest.main([__file__])
