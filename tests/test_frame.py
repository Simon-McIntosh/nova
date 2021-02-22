import pytest
import pandas
import numpy as np

from nova.electromagnetic.frame import Frame


def test_instance():
    frame = Frame()
    assert isinstance(frame, Frame)


def test_dataframe_subclass():
    assert issubclass(Frame, pandas.DataFrame)


def test_columns_update_error():
    with pytest.raises(IndexError):
        Frame(columns=['x', 'z'])


def test_index():
    frame = Frame(index=['Coil0', 'Coil1'])
    assert frame.index.to_list() == ['Coil0', 'Coil1']


def test_index_length_error():
    frame = Frame()
    with pytest.raises(IndexError):
        assert frame.add_frame(4, [5, 4, 6], 0.1, 0.3, name=['1, 2'])


def test_required_columns():
    frame = Frame(metadata={'Required': ['x', 'z']})
    assert frame.metadata['required'] == ['x', 'z']


def test_required_add_frame():
    frame = Frame(metadata={'Required': ['x', 'z']})
    frame.add_frame(1, 2)


def test_required_add_frame_error():
    frame = Frame(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert frame.add_frame(1, 2, 3)


def test_reset_metadata_attribute():
    frame = Frame(metadata={'additional': []})
    assert frame.metadata['additional'] == []


def test_frame_index():
    frame = Frame(metadata={'Required': ['x', 'z'], 'Additional': []})
    frame.add_frame(0, 1)
    assert list(frame.columns) == ['x', 'z']


def test_frame_index_mpc():
    frame = Frame(metadata={'Required': ['x', 'z'], 'Additional': []})
    frame.add_frame(0, 1, mpc=True)
    assert list(frame.columns) == ['x', 'z', 'mpc']


def test_data_init():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    frame = Frame(data, metadata={'Required': ['x', 'z']})
    assert len(frame) == 3


def test_data_init_mpc_true():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'mpc': True})
    frame = Frame(data, metadata={'Required': ['x', 'z'],
                                  'frame': {'name': 'Coil', 'delim': ''}})
    assert list(frame.mpc) == ['', ('Coil0', 1.0), ('Coil0', 1.0)]


def test_add_coil_mpc_default_false():
    frame = Frame({'x': [1, 3], 'z': 0}, mpc=False,
                  metadata={'Required': ['x', 'z'], 'Additional': [],
                            'frame': {'name': 'Coil', 'delim': ''}})
    frame.add_frame(4, [7, 8], mpc=True)
    assert list(frame.mpc) == ['', '', '', ('Coil2', 1.0)]


def test_add_coil_mpc_default_true():
    frame = Frame({'x': [1, 3], 'z': 0}, mpc=True,
                  metadata={'Required': ['x', 'z'], 'Additional': [],
                            'frame': {'name': 'Coil', 'delim': ''}})
    frame.add_frame(4, [7, 8], mpc=True)
    assert list(frame.mpc) == ['', ('Coil0', 1.0), '', ('Coil2', 1.0)]


def test_add_coil_mpc_default_float():
    frame = Frame({'x': [1, 3], 'z': 0}, mpc=0.5,
                  metadata={'Required': ['x', 'z'], 'Additional': [],
                            'frame': {'name': 'Coil', 'delim': ''}})
    frame.add_frame(4, [7, 8], mpc=True)
    assert list(frame.mpc) == ['', ('Coil0', 0.5), '', ('Coil2', 1.0)]


def test_default_mpc_true():
    frame = Frame(mpc=True, metadata={'Additional': ['mpc']})
    frame.add_frame(4, [5, 7, 12], name='coil1')
    assert list(frame.mpc) == ['', ('coil1', 1.0), ('coil1', 1.0)]


def test_data_init_mpc_false():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'mpc': False})
    frame = Frame(data, metadata={'Required': ['x', 'z'],
                                  'Additional': ['mpc'],
                                  'frame': {'name': 'Coil', 'delim': ''}})
    unset = [mpc == '' for mpc in frame.mpc]
    assert np.array(unset).all()


def test_data_init_required_error():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    with pytest.raises(IndexError):
        Frame(data, metadata={'Required': ['x', 'z', 'dl', 'dt']})


def test_data_init_additional_pass():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    frame = Frame(data, metadata={'Required': ['x', 'z'],
                                  'Additional': ['rms']})
    assert list(frame.columns) == ['x', 'z', 'rms']


def test_attribute_metadata_replace():
    frame = Frame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    frame.add_frame(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    frame = Frame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        frame.add_frame(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    frame = Frame({'x': 3, 'z': [4]}, metadata={
        'Required': ['x', 'z'], 'Additional': ['x', 'rms']})
    assert list(frame.columns) == ['x', 'z', 'rms']


def non_default_metaframe_frame_error():
    with pytest.raises(IndexError):
        Frame(metadata={'frame': {'link': False}})


def test_exclude_required_error():
    with pytest.raises(IndexError):
        Frame(metadata={'required': ['x', 'z'], 'Exclude': ['x']})


def test_exclude():
    frame = Frame(metadata={'required': ['x', 'z'],
                            'additional': ['rms', 'poly'],
                            'exclude': ['poly']})
    assert list(frame.metaframe.columns) == ['x', 'z', 'rms']


if __name__ == '__main__':

    pytest.main([__file__])
