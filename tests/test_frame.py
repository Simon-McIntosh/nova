import pytest
import pandas

from nova.electromagnetic.frame import Frame


def test_instance():
    frame = Frame()
    assert isinstance(frame, Frame)


def test_dataframe_subclass():
    assert issubclass(Frame, pandas.DataFrame)


def test_columns_extend_additional():
    frame = Frame(metadata={'Required': ['x', 'z'], 'Additional': []},
                  columns=['x', 'z', 'rms'])
    assert frame.columns.to_list() == ['x', 'z', 'rms']


def test_columns():
    frame = Frame(metadata={'Required': ['x', 'z'],
                            'Additional': ['rms']})
    frame.add_frame(2, [5, 6, 7])
    frame = Frame(frame, columns=['x', 'z'])
    print(frame)
    assert frame.columns.to_list() == ['x', 'z']


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


def test_frame_columns_multipoint():
    frame = Frame(metadata={'Required': ['x', 'z'], 'Additional': []})
    frame.add_frame(0, 1, link=True)
    assert list(frame.columns) == ['x', 'z', 'link', 'factor', 'reference']


def test_data_init():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    frame = Frame(data, metadata={'Required': ['x', 'z']})
    assert len(frame) == 3


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


def non_default_metaframe_index_error():
    with pytest.raises(IndexError):
        Frame(metadata={'index': {'link': False}})


def test_exclude_required_error():
    with pytest.raises(IndexError):
        Frame(metadata={'required': ['x', 'z'], 'Exclude': ['x']})


def test_exclude():
    frame = Frame(metadata={'Required': ['x', 'z'],
                            'Additional': ['rms', 'section'],
                            'Exclude': ['section']})
    assert list(frame.metaframe.columns) == ['x', 'z', 'rms']


def test_warn_new_attribute():
    frame = Frame({'x': [3, 4], 'z': 0}, metadata={'Required': ['x', 'z']})
    with pytest.warns(UserWarning,
                      match='Pandas doesn\'t allow columns to be created '
                            'via a new attribute name'):
        frame.Ic = [1, 2]


def test_restrict_lock():
    frame = Frame(metadata={'Required': ['x'], 'Restrict': ['x']})
    with pytest.raises(PermissionError):
        frame.metaarray.check_lock('x')


def test_restricted_variable_access_error():
    frame = Frame(metadata={'Required': ['x'], 'Restrict': ['Ic'],
                            'Additional': ['Ic']})
    frame.add_frame(5)
    with pytest.raises(PermissionError):
        frame.Ic = 6


if __name__ == '__main__':

    test_columns()
    #pytest.main([__file__])
