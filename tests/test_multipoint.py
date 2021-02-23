
import pytest
import numpy as np
import pandas

from nova.electromagnetic.frame import Frame


def test_enable_key_attribute_true():
    frame = Frame({'mpc': [True]}, metadata={'Required': []})
    assert frame.multipoint.enable


def test_generate_single():
    frame = Frame({'mpc': [True]}, metadata={'Required': []})
    assert frame.iloc[0].to_list() == ['', 1.0]


def test_generate_single_float():
    frame = Frame({'mpc': [-0.3]}, metadata={'Required': []})
    assert frame.iloc[0].to_list() == ['', 1.0]


def test_generate_multi():
    frame = Frame({'x': range(3), 'mpc': True}, metadata={
        'Required': ['x'], 'frame': {'name': 'coil1'}})
    assert frame.mpc.to_list() == ['', 'coil1', 'coil1']


def test_data_init_multipoint_true():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'mpc': True})
    frame = Frame(data, metadata={'Required': ['x', 'z'],
                                  'frame': {'name': 'Coil0'}})
    assert frame.mpc.to_list() == ['', 'Coil0', 'Coil0'] and \
        frame.factor.to_list() == [1, 1, 1]


def test_add_coil_multipoint_default_false():
    frame = Frame({'x': [1, 3], 'z': 0}, mpc=False,
                  metadata={'Required': ['x', 'z'], 'Additional': [],
                            'frame': {'name': 'Coil0'}})
    frame.add_frame(4, [7, 8], mpc=True)
    assert frame.mpc.to_list() == ['', '', '', 'Coil2'] and \
        frame.factor.to_list() == [1, 1, 1, 1]


def test_add_coil_multipoint_default_true():
    frame = Frame({'x': [1, 3], 'z': 0}, mpc=True,
                  metadata={'Required': ['x', 'z'], 'Additional': [],
                            'frame': {'name': 'Coil0'}})
    frame.add_frame(4, [7, 8], mpc=True)
    assert frame.mpc.to_list() == ['', 'Coil0', '', 'Coil2'] and \
        frame.factor.to_list() == [1, 1, 1, 1]


def test_add_coil_multipoint_default_float():
    frame = Frame({'x': [1, 3], 'z': 0}, mpc=0.5,
                  metadata={'Required': ['x', 'z'], 'Additional': [],
                            'frame': {'name': 'Coil0'}})
    frame.add_frame(4, [7, 8], mpc=True)
    assert frame.mpc.to_list() == ['', 'Coil0', '', 'Coil2'] and \
        frame.factor.to_list() == [1, 0.5, 1, 1]


def test_default_multipoint_true():
    frame = Frame(mpc=True, metadata={'Required': ['x', 'z'],
                                      'Additional': ['mpc']})
    frame.add_frame(4, [5, 7, 12], name='coil1')
    assert frame.mpc.to_list() == ['', 'coil1', 'coil1'] and \
        frame.factor.to_list() == [1, 1, 1]


def test_data_init_multipoint_false():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'mpc': False})
    frame = Frame(data, metadata={'Required': ['x', 'z'],
                                  'Additional': ['mpc'],
                                  'frame': {'name': 'Coil0'}})
    unset = [mpc == '' for mpc in frame.mpc]
    assert np.array(unset).all()


if __name__ == '__main__':

    pytest.main([__file__])
