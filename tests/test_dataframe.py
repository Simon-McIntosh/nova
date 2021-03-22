import pytest
import pandas

from nova.electromagnetic.dataframe import DataFrame


def test_instance():
    dataframe = DataFrame()
    assert isinstance(dataframe, DataFrame)


def test_init_metadata():
    dataframe = DataFrame(Required=['x', 'z'], Additional=[])
    assert dataframe.metaframe.required == ['x', 'z']


def test_dataframe_subclass():
    assert issubclass(DataFrame, pandas.DataFrame)


def test_columns_extend_additional():
    dataframe = DataFrame(Required=['x', 'z'],
                          Additional=[], columns=['x', 'z', 'rms'])
    assert dataframe.columns.to_list() == ['x', 'z', 'rms']


def test_index():
    dataframe = DataFrame(index=['Coil0', 'Coil1'])
    assert dataframe.index.to_list() == ['Coil0', 'Coil1']


def test_reindex():
    dataframe = DataFrame({'x': range(10)}, metadata={'Required': ['x']})
    dataframe = DataFrame(dataframe, index=['Coil2', 'Coil7', 'Coil9'])
    assert dataframe.x.to_list() == [2, 7, 9]


def test_required_columns():
    dataframe = DataFrame(metadata={'Required': ['x', 'z']})
    assert dataframe.metadata['required'] == ['x', 'z']


def test_reset_metadata_attribute():
    dataframe = DataFrame(metadata={'additional': []})
    assert dataframe.metadata['additional'] == []


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
