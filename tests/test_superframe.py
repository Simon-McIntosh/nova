import pytest
import pandas

from nova.electromagnetic.superframe import SuperFrame


def test_instance():
    superframe = SuperFrame()
    assert isinstance(superframe, SuperFrame)


def test_dataframe_subclass():
    assert issubclass(SuperFrame, pandas.DataFrame)


def test_columns_extend_additional():
    superframe = SuperFrame(Required=['x', 'z'], Additional=[],
                            columns=['x', 'z', 'rms'])
    assert superframe.columns.to_list() == ['x', 'z', 'rms']


def test_columns():
    superframe = SuperFrame(metadata={'Required': ['x', 'z'],
                            'Additional': ['rms']})
    superframe.add_frame(2, [5, 6, 7], rms=5)
    superframe = SuperFrame(superframe, columns=['x', 'z'])
    assert superframe.columns.to_list() == ['x', 'z']


def test_columns_metaframe_update():
    superframe = SuperFrame(metadata={'Required': ['x', 'z', 'dl'],
                            'Additional': ['rms']},
                  columns=['x', 'dt'])
    superframe.add_frame(4, dt=[5, 7, 12])
    superframe = SuperFrame(superframe, columns=['x', 'dt', 'dl'])
    assert superframe.columns.to_list() == ['x', 'dt', 'dl']


def test_index():
    superframe = SuperFrame(index=['Coil0', 'Coil1'])
    assert superframe.index.to_list() == ['Coil0', 'Coil1']


def test_reindex():
    superframe = SuperFrame({'x': range(10)}, metadata={'Required': ['x']})
    superframe = SuperFrame(superframe, index=['Coil2', 'Coil7', 'Coil9'])
    assert superframe.x.to_list() == [2, 7, 9]


def test_index_length_error():
    superframe = SuperFrame()
    with pytest.raises(IndexError):
        assert superframe.add_frame(4, [5, 4, 6], 0.1, 0.3, name=['1, 2'])


def test_required_columns():
    superframe = SuperFrame(metadata={'Required': ['x', 'z']})
    assert superframe.metadata['required'] == ['x', 'z']


def test_required_add_frame():
    superframe = SuperFrame(metadata={'Required': ['x', 'z']})
    superframe.add_frame(1, 2)


def test_required_add_frame_error():
    superframe = SuperFrame(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert superframe.add_frame(1, 2, 3)


def test_reset_metadata_attribute():
    superframe = SuperFrame(metadata={'additional': []})
    assert superframe.metadata['additional'] == []


def test_frame_index():
    superframe = SuperFrame(metadata={'Required': ['x', 'z'], 'Additional': []})
    superframe.add_frame(0, 1)
    assert list(superframe.columns) == ['x', 'z']


def test_frame_columns_multipoint():
    superframe = SuperFrame(metadata={'Required': ['x', 'z'], 'Additional': []})
    superframe.add_frame(0, 1, link=True)
    assert list(superframe.columns) == ['x', 'z', 'link', 'factor', 'ref', 'subref']


def test_data_init():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    superframe = SuperFrame(data, metadata={'Required': ['x', 'z']})
    assert len(superframe) == 3


def test_data_init_required():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    superframe = SuperFrame(data, metadata={'Required': ['x', 'z', 'dl', 'dt']})
    assert superframe.columns.to_list() == ['x', 'z']


def test_data_init_additional():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'dl': 0.3})
    superframe = SuperFrame(data, metadata={'Required': ['x', 'z'],
                                  'Additional': ['rms']})
    assert list(superframe.columns) == ['x', 'z', 'dl']


def test_attribute_metadata_replace():
    superframe = SuperFrame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    superframe.add_frame(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    superframe = SuperFrame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        superframe.add_frame(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    superframe = SuperFrame(Required=['x', 'z'], Additional=['x', 'rms'])
    superframe.add_frame(3, 4)
    assert list(superframe.columns) == ['x', 'z', 'rms']


def non_default_metaframe_index_error():
    with pytest.raises(IndexError):
        SuperFrame(metadata={'index': {'link': False}})


def test_exclude_required_error():
    with pytest.raises(IndexError):
        SuperFrame(metadata={'required': ['x', 'z'], 'Exclude': ['x']})


def test_exclude():
    superframe = SuperFrame(metadata={'Required': ['x', 'z'],
                                      'Additional': ['rms', 'section'],
                                      'Exclude': ['section']})
    assert list(superframe.metaframe.columns) == ['x', 'z', 'rms']


def test_warn_new_attribute():
    superframe = SuperFrame({'x': [3, 4], 'z': 0}, Required=['x', 'z'],
                            Subspace=[])
    with pytest.warns(UserWarning,
                      match='Pandas doesn\'t allow columns to be created '
                            'via a new attribute name'):
        superframe.Ic = [1, 2]


if __name__ == '__main__':

    pytest.main([__file__])
