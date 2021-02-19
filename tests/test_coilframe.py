import pytest
import pandas

from nova.electromagnetic.coilframe import CoilFrame


def test_instance():
    frame = CoilFrame()
    assert isinstance(frame, CoilFrame)


def test_dataframe_subclass():
    assert issubclass(CoilFrame, pandas.DataFrame)


def test_columns():
    frame = CoilFrame(columns=['x', 'z'])
    assert sorted(frame.columns) == sorted(['x', 'z'])


def test_index():
    frame = CoilFrame(index=['Coil0', 'Coil1'])
    assert sorted(frame.index) == sorted(['Coil0', 'Coil1'])


def test_index_length_error():
    frame = CoilFrame()
    with pytest.raises(IndexError):
        assert frame.add_frame(4, [5, 4, 6], 0.1, 0.3, name=['1, 2'])


def test_required_columns():
    frame = CoilFrame(metadata={'Required': ['x', 'z']})
    assert frame.metadata['required'] == ['x', 'z']


def test_required_add_frame():
    frame = CoilFrame(metadata={'Required': ['x', 'z']})
    print('req', frame.metadata['required'])
    print(frame.attrs)
    frame.add_frame(1, 2)


def test_required_add_frame_error():
    frame = CoilFrame(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert frame.add_frame(1, 2, 3)


def test_reset_metadata_attribute():
    frame = CoilFrame(metadata={'additional': []})
    assert frame.metadata['additional'] == []


def test_frame_index():
    frame = CoilFrame(metadata={'Required': ['x', 'z'], 'additional': []})
    frame.add_frame(0, 1)
    assert list(frame.columns) == ['x', 'z']


def test_data_init():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    frame = CoilFrame(data, metadata={'Required': ['x', 'z']})
    assert frame.coil_number == 3


def test_data_init_required_error():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    with pytest.raises(IndexError):
        CoilFrame(data, metadata={'Required': ['x', 'z', 'dl', 'dt']})


def test_data_init_additional_pass():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    frame = CoilFrame(data, metadata={'Required': ['x', 'z'],
                                      'Additional': ['rms']})
    assert list(frame.columns) == ['x', 'z', 'rms']


def test_upper_attribute_metadata_replace():
    CoilFrame({'x': 3, 'z': [4]}, metadata={'Required': ['x', 'z']})


def test_upper_attribute_metadata_replace_error():
    with pytest.raises(IndexError):
        CoilFrame({'x': 3, 'z': [4]}, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    frame = CoilFrame({'x': 3, 'z': [4]}, metadata={
        'Required': ['x', 'z'], 'Additional': ['x', 'rms']})
    assert list(frame.columns) == ['x', 'z', 'rms']


if __name__ == '__main__':

    pytest.main([__file__])
