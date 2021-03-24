
import pytest

from nova.electromagnetic.frameset import FrameSet


def test_drop():
    frameset = FrameSet(Required=['x', 'z'], label='PF')
    frameset.insert(2, range(2))
    frameset.insert(1, range(3), link=True)
    frameset.insert(3, 7)
    frameset.drop('PF4')
    frameset.drop(['PF0', 'PF1'])
    assert frameset.index.to_list() == ['PF2', 'PF3', 'PF5']
    assert frameset.loc[:, 'ref'].to_list() == [0, 0, 2]
    assert frameset.loc[:, 'subref'].to_list() == [0, 0, 1]


def test_delim():
    frameset = FrameSet(Required=['x', 'z'], label='PF', delim='_')
    frameset.insert(2, range(2))
    assert frameset.index.to_list() == ['PF_0', 'PF_1']


def test_columns():
    frameset = FrameSet(metadata={'Required': ['x', 'z'],
                                  'Additional': ['rms']})
    frameset.insert(2, [5, 6, 7], rms=5)
    frameset = FrameSet(frameset, columns=['x', 'z'])
    assert frameset.columns.to_list() == ['x', 'z']


def test_columns_metaframe_update():
    frameset = FrameSet(metadata={'Required': ['x', 'z', 'dl'],
                                  'Additional': ['rms']},
                        columns=['x', 'dt'])
    frameset.insert(4, dt=[5, 7, 12])
    frameset = FrameSet(frameset, columns=['x', 'dt', 'dl'])
    assert frameset.columns.to_list() == ['x', 'dt', 'dl']


def test_index_length_error():
    frameset = FrameSet()
    with pytest.raises(IndexError):
        assert frameset.insert(4, [5, 4, 6], 0.1, 0.3, name=['1, 2'])


def test_required_add_frame():
    frameset = FrameSet(metadata={'Required': ['x', 'z']})
    frameset.insert(1, 2)


def test_required_add_frame_error():
    frameset = FrameSet(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert frameset.insert(1, 2, 3)


def test_attribute_metadata_replace():
    frameset = FrameSet(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    frameset.insert(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    frameset = FrameSet(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        frameset.insert(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    frameset = FrameSet(Required=['x', 'z'], Additional=['x', 'rms'])
    frameset.insert(3, 4)
    assert list(frameset.columns) == ['x', 'z', 'rms']


def test_frame_columns():
    frameset = FrameSet(metadata={'Required': ['x', 'z'], 'Additional': []})
    frameset.insert(0, 1)
    assert list(frameset.columns) == ['x', 'z']


def test_Ic_unset():
    frameset = FrameSet(Required=['x'])
    frameset.insert([4, 5], It=6.5)
    assert frameset.Ic.to_list() == [6.5, 6.5]


def test_data_It_Ic_unset():
    frameset = FrameSet({'x': [1, 2], 'It': 5, 'Nt': 2.5}, Required=['x'])
    assert frameset.Ic.to_list() == [2, 2]


def test_data_It_Ic_set():
    frameset = FrameSet({'x': [1, 2], 'It': 5, 'Ic': 10, 'Nt': 2.5},
                        Required=['x'])
    assert frameset.Ic.to_list() == [10, 10]
    assert frameset.It.to_list() == [25, 25]


def test_Ic_unset_Additional():
    frameset = FrameSet(Required=['x', 'z'], Additional=['Ic'])
    frameset.insert(4, range(2), It=5)
    assert frameset.Ic.to_list() == [5, 5]


def test_init_insert_label():
    frameset = FrameSet(Required=['x'], label='CS')
    frameset.insert(range(3))
    assert frameset.index.to_list() == ['CS0', 'CS1', 'CS2']


def test_insert_insert_label_switch():
    frameset = FrameSet(Required=['x', 'z'], label='CS')
    frameset.insert(4, 1, label='PF')
    frameset.insert(range(2), 1, offset=7)
    frameset.insert(2, range(2), label='PF')
    frameset.insert(range(2), 1)
    assert frameset.index.to_list() == \
        ['PF0', 'CS7', 'CS8', 'PF1', 'PF2', 'CS9', 'CS10']


if __name__ == '__main__':

    pytest.main([__file__])
