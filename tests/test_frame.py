import pytest
import pandas

from nova.electromagnetic.frame import Frame


def test_instance():
    frame = Frame()
    assert isinstance(frame, Frame)


def test_Ic_unset():
    frame = Frame(Required=['x'])
    frame.add_frame([4, 5], It=6.5)
    assert frame.Ic.to_list() == [6.5, 6.5]


def test_data_It_Ic_unset():
    frame = Frame({'x': [1, 2], 'It': 5, 'Nt': 2.5}, Required=['x'])
    assert frame.Ic.to_list() == [2, 2]


def test_data_It_Ic_set():
    frame = Frame({'x': [1, 2], 'It': 5, 'Ic': 10, 'Nt': 2.5}, Required=['x'])
    assert frame.Ic.to_list() == [10, 10]
    assert frame.It.to_list() == [25, 25]


def test_Ic_unset_Additional():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.add_frame(4, range(2), It=5)
    assert frame.Ic.to_list() == [5, 5]


def test_dataframe_subclass():
    assert issubclass(Frame, pandas.DataFrame)


def test_columns_extend_additional():
    frame = Frame(Required=['x', 'z'],
                  Additional=[], columns=['x', 'z', 'rms'])
    assert frame.columns.to_list() == ['x', 'z', 'rms']


def test_columns():
    frame = Frame(metadata={'Required': ['x', 'z'],
                            'Additional': ['rms']})
    frame.add_frame(2, [5, 6, 7], rms=5)
    frame = Frame(frame, columns=['x', 'z'])
    assert frame.columns.to_list() == ['x', 'z']


def test_columns_metaframe_update():
    frame = Frame(metadata={'Required': ['x', 'z', 'dl'],
                            'Additional': ['rms']},
                  columns=['x', 'dt'])
    frame.add_frame(4, dt=[5, 7, 12])
    frame = Frame(frame, columns=['x', 'dt', 'dl'])
    assert frame.columns.to_list() == ['x', 'dt', 'dl']


def test_index():
    frame = Frame(index=['Coil0', 'Coil1'])
    assert frame.index.to_list() == ['Coil0', 'Coil1']


def test_reindex():
    frame = Frame({'x': range(10)}, metadata={'Required': ['x']})
    frame = Frame(frame, index=['Coil2', 'Coil7', 'Coil9'])
    assert frame.x.to_list() == [2, 7, 9]


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
    assert list(frame.columns) == ['x', 'z', 'link', 'factor', 'ref', 'subref']


def test_data_init():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    frame = Frame(data, metadata={'Required': ['x', 'z']})
    assert len(frame) == 3


def test_data_init_required():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    frame = Frame(data, metadata={'Required': ['x', 'z', 'dl', 'dt']})
    assert frame.columns.to_list() == ['x', 'z']


def test_data_init_additional():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'dl': 0.3})
    frame = Frame(data, columns=['x', 'z'], Additional=['rms'])
    assert list(frame.columns) == ['x', 'z', 'rms']


def test_attribute_metadata_replace():
    frame = Frame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    frame.add_frame(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    frame = Frame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        frame.add_frame(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    frame = Frame(Required=['x', 'z'], Additional=['x', 'rms'])
    frame.add_frame(3, 4)
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
    frame = Frame({'x': [3, 4], 'z': 0}, Required=['x', 'z'],
                  Subspace=[])
    with pytest.warns(UserWarning,
                      match='Pandas doesn\'t allow columns to be created '
                            'via a new attribute name'):
        frame.Ic = [1, 2]


if __name__ == '__main__':

    pytest.main([__file__])
