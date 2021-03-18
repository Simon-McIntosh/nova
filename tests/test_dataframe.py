import pytest
import pandas

from nova.electromagnetic.dataframe import DataFrame


def test_instance():
    dataframe = DataFrame()
    assert isinstance(dataframe, DataFrame)


def test_init_metadata():
    dataframe = DataFrame(Required=['x', 'z'], Additional=[])
    assert dataframe.metaframe.required == ['x', 'z']


def test_Ic_unset():
    dataframe = DataFrame(Required=['x'])
    dataframe.add_frame([4, 5], It=6.5)
    assert dataframe.Ic.to_list() == [6.5, 6.5]


def test_data_It_Ic_unset():
    dataframe = DataFrame({'x': [1, 2], 'It': 5, 'Nt': 2.5}, Required=['x'])
    assert dataframe.Ic.to_list() == [2, 2]


def test_data_It_Ic_set():
    dataframe = DataFrame({'x': [1, 2], 'It': 5, 'Ic': 10, 'Nt': 2.5},
                          Required=['x'])
    assert dataframe.Ic.to_list() == [10, 10]
    assert dataframe.It.to_list() == [25, 25]


def test_Ic_unset_Additional():
    dataframe = DataFrame(Required=['x', 'z'], Additional=['Ic'])
    dataframe.add_frame(4, range(2), It=5)
    assert dataframe.Ic.to_list() == [5, 5]


def test_dataframe_subclass():
    assert issubclass(DataFrame, pandas.DataFrame)


def test_columns_extend_additional():
    dataframe = DataFrame(Required=['x', 'z'],
                          Additional=[], columns=['x', 'z', 'rms'])
    assert dataframe.columns.to_list() == ['x', 'z', 'rms']


def test_columns():
    dataframe = DataFrame(metadata={'Required': ['x', 'z'],
                                    'Additional': ['rms']})
    dataframe.add_frame(2, [5, 6, 7], rms=5)
    dataframe = DataFrame(dataframe, columns=['x', 'z'])
    assert dataframe.columns.to_list() == ['x', 'z']


def test_columns_metaframe_update():
    dataframe = DataFrame(metadata={'Required': ['x', 'z', 'dl'],
                                    'Additional': ['rms']},
                          columns=['x', 'dt'])
    dataframe.add_frame(4, dt=[5, 7, 12])
    dataframe = DataFrame(dataframe, columns=['x', 'dt', 'dl'])
    assert dataframe.columns.to_list() == ['x', 'dt', 'dl']


def test_index():
    dataframe = DataFrame(index=['Coil0', 'Coil1'])
    assert dataframe.index.to_list() == ['Coil0', 'Coil1']


def test_reindex():
    dataframe = DataFrame({'x': range(10)}, metadata={'Required': ['x']})
    dataframe = DataFrame(dataframe, index=['Coil2', 'Coil7', 'Coil9'])
    assert dataframe.x.to_list() == [2, 7, 9]


def test_index_length_error():
    dataframe = DataFrame()
    with pytest.raises(IndexError):
        assert dataframe.add_frame(4, [5, 4, 6], 0.1, 0.3, name=['1, 2'])


def test_required_columns():
    dataframe = DataFrame(metadata={'Required': ['x', 'z']})
    assert dataframe.metadata['required'] == ['x', 'z']


def test_required_add_frame():
    dataframe = DataFrame(metadata={'Required': ['x', 'z']})
    dataframe.add_frame(1, 2)


def test_required_add_frame_error():
    dataframe = DataFrame(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert dataframe.add_frame(1, 2, 3)


def test_reset_metadata_attribute():
    dataframe = DataFrame(metadata={'additional': []})
    assert dataframe.metadata['additional'] == []


def test_frame_index():
    dataframe = DataFrame(metadata={'Required': ['x', 'z'], 'Additional': []})
    dataframe.add_frame(0, 1)
    assert list(dataframe.columns) == ['x', 'z']


def test_frame_columns_multipoint():
    dataframe = DataFrame(metadata={'Required': ['x', 'z'], 'Additional': []})
    dataframe.add_frame(0, 1, link=True)
    assert list(dataframe.columns) == ['x', 'z', 'link', 'factor',
                                       'ref', 'subref']


def test_data_init():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    dataframe = DataFrame(data, metadata={'Required': ['x', 'z']})
    assert len(dataframe) == 3


def test_data_init_required():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8]})
    dataframe = DataFrame(data, metadata={'Required': ['x', 'z', 'dl', 'dt']})
    assert dataframe.columns.to_list() == ['x', 'z']


def test_data_init_additional():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'dl': 0.3})
    dataframe = DataFrame(data, metadata={'Required': ['x', 'z'],
                                          'Additional': ['rms']})
    assert list(dataframe.columns) == ['x', 'z', 'dl', 'rms']


def test_attribute_metadata_replace():
    dataframe = DataFrame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    dataframe.add_frame(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    dataframe = DataFrame(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        dataframe.add_frame(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    dataframe = DataFrame(Required=['x', 'z'], Additional=['x', 'rms'])
    dataframe.add_frame(3, 4)
    assert list(dataframe.columns) == ['x', 'z', 'rms']


def non_default_metaframe_index_error():
    with pytest.raises(IndexError):
        DataFrame(metadata={'index': {'link': False}})


def test_exclude_required_error():
    with pytest.raises(IndexError):
        DataFrame(metadata={'required': ['x', 'z'], 'Exclude': ['x']})


def test_exclude():
    dataframe = DataFrame(metadata={'Required': ['x', 'z'],
                                    'Additional': ['rms', 'section'],
                                    'Exclude': ['section']})
    assert list(dataframe.metaframe.columns) == ['x', 'z', 'rms']


def test_warn_new_attribute():
    dataframe = DataFrame({'x': [3, 4], 'z': 0}, Required=['x', 'z'],
                          Subspace=[])
    with pytest.warns(UserWarning,
                      match='Pandas doesn\'t allow columns to be created '
                            'via a new attribute name'):
        dataframe.Ic = [1, 2]


if __name__ == '__main__':

    pytest.main([__file__])
