
import pytest

from nova.electromagnetic.framearray import FrameArray


def test_drop():
    framearray = FrameArray(Required=['x', 'z'], label='PF')
    framearray.insert(2, range(2))
    framearray.insert(1, range(3), link=True)
    framearray.insert(3, 7)
    framearray.drop('PF4')
    framearray.drop(['PF0', 'PF1'])
    assert framearray.index.to_list() == ['PF2', 'PF3', 'PF5']
    assert framearray.loc[:, 'ref'].to_list() == [0, 0, 2]
    assert framearray.loc[:, 'subref'].to_list() == [0, 0, 1]


def test_delim():
    framearray = FrameArray(Required=['x', 'z'], label='PF', delim='_')
    framearray.insert(2, range(2))
    assert framearray.index.to_list() == ['PF_0', 'PF_1']


def test_columns():
    framearray = FrameArray(metadata={'Required': ['x', 'z'],
                                      'Additional': ['rms']})
    framearray.insert(2, [5, 6, 7], rms=5)
    framearray = FrameArray(framearray, columns=['x', 'z'])
    assert framearray.columns.to_list() == ['x', 'z']


def test_columns_metaframe_update():
    framearray = FrameArray(metadata={'Required': ['x', 'z', 'dl'],
                                      'Additional': ['rms']},
                            columns=['x', 'dt'])
    framearray.insert(4, dt=[5, 7, 12])
    framearray = FrameArray(framearray, columns=['x', 'dt', 'dl'])
    assert framearray.columns.to_list() == ['x', 'dt', 'dl']


def test_index_length_error():
    framearray = FrameArray()
    with pytest.raises(IndexError):
        assert framearray.insert(4, [5, 4, 6], 0.1, 0.3, name=['1, 2'])


def test_required_add_frame():
    framearray = FrameArray(metadata={'Required': ['x', 'z']})
    framearray.insert(1, 2)


def test_required_add_frame_error():
    framearray = FrameArray(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert framearray.insert(1, 2, 3)


def test_attribute_metadata_replace():
    framearray = FrameArray(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    framearray.insert(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    framearray = FrameArray(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        framearray.insert(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    framearray = FrameArray(Required=['x', 'z'], Additional=['x', 'rms'])
    framearray.insert(3, 4)
    assert list(framearray.columns) == ['x', 'z', 'rms']


def test_frame_columns():
    framearray = FrameArray(metadata={'Required': ['x', 'z'],
                                      'Additional': []})
    framearray.insert(0, 1)
    assert list(framearray.columns) == ['x', 'z']


def test_Ic_unset():
    framearray = FrameArray(Required=['x'])
    framearray.insert([4, 5], It=6.5)
    assert framearray.Ic.to_list() == [6.5, 6.5]


def test_data_It_Ic_unset():
    framearray = FrameArray({'x': [1, 2], 'It': 5, 'nturn': 2.5},
                            Required=['x'])
    assert framearray.Ic.to_list() == [2, 2]


def test_data_It_Ic_set():
    framearray = FrameArray({'x': [1, 2], 'It': 5, 'Ic': 10, 'nturn': 2.5},
                            Required=['x'])
    assert framearray.Ic.to_list() == [10, 10]
    assert framearray.It.to_list() == [25, 25]


def test_Ic_unset_Additional():
    framearray = FrameArray(Required=['x', 'z'], Additional=['Ic'])
    framearray.insert(4, range(2), It=5)
    assert framearray.Ic.to_list() == [5, 5]


def test_init_insert_label():
    framearray = FrameArray(Required=['x'], label='CS')
    framearray.insert(range(3))
    assert framearray.index.to_list() == ['CS0', 'CS1', 'CS2']


def test_insert_insert_label_switch():
    framearray = FrameArray(Required=['x', 'z'], label='CS')
    framearray.insert(4, 1, label='PF')
    framearray.insert(range(2), 1, offset=7)
    framearray.insert(2, range(2), label='PF')
    framearray.insert(range(2), 1)
    assert framearray.index.to_list() == \
        ['PF0', 'CS7', 'CS8', 'PF1', 'PF2', 'CS9', 'CS10']


def test_reindex():
    framearray = FrameArray({'x': range(10)}, Required=['x'])
    framearray = FrameArray(framearray, index=['Coil2', 'Coil7', 'Coil9'])
    assert framearray.x.to_list() == [2, 7, 9]


def test_named_multi_insert():
    framearray = FrameArray(required=['x', 'z'], label='CS', offset=11)
    framearray.insert([-4, -5], 1, Ic=6.5, name='PF4')
    framearray.insert([-4, -5], 2, Ic=6.5)
    framearray.insert([-4, -5], 3, Ic=6.5, name='PF0')
    assert framearray.index.to_list() == \
        ['PF4', 'PF5', 'CS11', 'CS12', 'PF6', 'PF7']


def test_named_insert_no_number():
    framearray = FrameArray(required=['x', 'z'], label='CS', offset=11)
    framearray.insert([-4, -5], 1, Ic=6.5, name='PF')
    assert framearray.index.to_list() == ['PF11', 'PF12']


def test_named_insert_number_not_trailing():
    framearray = FrameArray(required=['x', 'z'], label='CS', offset=11)
    framearray.insert([-4, -5], 1, Ic=6.5, name='CS_3_PF')
    assert framearray.index.to_list() == ['CS_3_PF11', 'CS_3_PF12']


def test_labelled_insert_offset():
    framearray = FrameArray(required=['x', 'z'], label='CS', offset=7)
    framearray.insert([-4, -5], 1, Ic=6.5, label='PF')
    framearray.insert([-4, -5], 2, Ic=6.5)
    assert framearray.index.to_list() == ['PF7', 'PF8', 'CS7', 'CS8']


if __name__ == '__main__':

    pytest.main([__file__])
