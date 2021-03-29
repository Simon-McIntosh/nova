
import pytest

from nova.electromagnetic.unitset import UnitSet


def test_drop():
    unitset = UnitSet(Required=['x', 'z'], label='PF')
    unitset.insert(2, range(2))
    unitset.insert(1, range(3), link=True)
    unitset.insert(3, 7)
    unitset.drop('PF4')
    unitset.drop(['PF0', 'PF1'])
    assert unitset.index.to_list() == ['PF2', 'PF3', 'PF5']
    assert unitset.loc[:, 'ref'].to_list() == [0, 0, 2]
    assert unitset.loc[:, 'subref'].to_list() == [0, 0, 1]


def test_delim():
    unitset = UnitSet(Required=['x', 'z'], label='PF', delim='_')
    unitset.insert(2, range(2))
    assert unitset.index.to_list() == ['PF_0', 'PF_1']


def test_columns():
    unitset = UnitSet(metadata={'Required': ['x', 'z'],
                                'Additional': ['rms']})
    unitset.insert(2, [5, 6, 7], rms=5)
    unitset = UnitSet(unitset, columns=['x', 'z'])
    assert unitset.columns.to_list() == ['x', 'z']


def test_columns_metaframe_update():
    unitset = UnitSet(metadata={'Required': ['x', 'z', 'dl'],
                                'Additional': ['rms']},
                      columns=['x', 'dt'])
    unitset.insert(4, dt=[5, 7, 12])
    unitset = UnitSet(unitset, columns=['x', 'dt', 'dl'])
    assert unitset.columns.to_list() == ['x', 'dt', 'dl']


def test_index_length_error():
    unitset = UnitSet()
    with pytest.raises(IndexError):
        assert unitset.insert(4, [5, 4, 6], 0.1, 0.3, name=['1, 2'])


def test_required_add_frame():
    unitset = UnitSet(metadata={'Required': ['x', 'z']})
    unitset.insert(1, 2)


def test_required_add_frame_error():
    unitset = UnitSet(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert unitset.insert(1, 2, 3)


def test_attribute_metadata_replace():
    unitset = UnitSet(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    unitset.insert(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    unitset = UnitSet(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        unitset.insert(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    unitset = UnitSet(Required=['x', 'z'], Additional=['x', 'rms'])
    unitset.insert(3, 4)
    assert list(unitset.columns) == ['x', 'z', 'rms']


def test_frame_columns():
    unitset = UnitSet(metadata={'Required': ['x', 'z'], 'Additional': []})
    unitset.insert(0, 1)
    assert list(unitset.columns) == ['x', 'z']


def test_Ic_unset():
    unitset = UnitSet(Required=['x'])
    unitset.insert([4, 5], It=6.5)
    assert unitset.Ic.to_list() == [6.5, 6.5]


def test_data_It_Ic_unset():
    unitset = UnitSet({'x': [1, 2], 'It': 5, 'Nt': 2.5}, Required=['x'])
    assert unitset.Ic.to_list() == [2, 2]


def test_data_It_Ic_set():
    unitset = UnitSet({'x': [1, 2], 'It': 5, 'Ic': 10, 'Nt': 2.5},
                      Required=['x'])
    assert unitset.Ic.to_list() == [10, 10]
    assert unitset.It.to_list() == [25, 25]


def test_Ic_unset_Additional():
    unitset = UnitSet(Required=['x', 'z'], Additional=['Ic'])
    unitset.insert(4, range(2), It=5)
    assert unitset.Ic.to_list() == [5, 5]


def test_init_insert_label():
    unitset = UnitSet(Required=['x'], label='CS')
    unitset.insert(range(3))
    assert unitset.index.to_list() == ['CS0', 'CS1', 'CS2']


def test_insert_insert_label_switch():
    unitset = UnitSet(Required=['x', 'z'], label='CS')
    unitset.insert(4, 1, label='PF')
    unitset.insert(range(2), 1, offset=7)
    unitset.insert(2, range(2), label='PF')
    unitset.insert(range(2), 1)
    assert unitset.index.to_list() == \
        ['PF0', 'CS7', 'CS8', 'PF1', 'PF2', 'CS9', 'CS10']


def test_reindex():
    unitset = UnitSet({'x': range(10)}, Required=['x'])
    unitset = UnitSet(unitset, index=['Coil2', 'Coil7', 'Coil9'])
    assert unitset.x.to_list() == [2, 7, 9]


if __name__ == '__main__':

    pytest.main([__file__])
