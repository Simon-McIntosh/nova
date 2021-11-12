import pytest

from nova.electromagnetic.framelink import FrameLink


def test_instance():
    framelink = FrameLink()
    assert isinstance(framelink, FrameLink)


def test_data_It_Ic_unset_insert():
    framelink = FrameLink(Required=['x'])
    framelink.insert([1, 2], It=5, nturn=2.5)
    assert framelink.Ic.to_list() == [2, 2]


def test_data_It_Ic_set_insert():
    framelink = FrameLink(Required=['x'])
    framelink.insert([1, 2], It=5, Ic=10, nturn=2.5)
    assert framelink.Ic.to_list() == [10, 10]
    assert framelink.It.to_list() == [25, 25]


def test_data_It_Ic_unset_dict():
    framelink = FrameLink({'x': [1, 2], 'It': 5, 'nturn': 2.5},
                          Required=['x'])
    assert framelink.Ic.to_list() == [2, 2]


def test_data_It_Ic_set_dict():
    framelink = FrameLink({'x': [1, 2], 'It': 5, 'Ic': 10, 'nturn': 2.5},
                          Required=['x'])
    assert framelink.Ic.to_list() == [10, 10]
    assert framelink.It.to_list() == [25, 25]


def test_frame_columns_multipoint():
    framelink = FrameLink(metadata={'Required': ['x', 'z'], 'Additional': []})
    framelink.insert(0, 1, link=True)
    assert list(framelink.columns) == ['x', 'z', 'link', 'factor', 'ref',
                                       'subref']


def test_drop_multipoint():
    framelink = FrameLink(Required=['x', 'z'], label='PF')
    framelink.insert(2, range(2))
    framelink.insert(1, range(3), link=True)
    framelink.insert(3, 7)
    framelink.drop('PF4')
    framelink.drop(['PF0', 'PF1'])
    assert framelink.index.to_list() == ['PF2', 'PF3', 'PF5']
    assert framelink.loc[:, 'ref'].to_list() == [0, 0, 2]
    assert framelink.loc[:, 'subref'].to_list() == [0, 0, 1]


def test_delim():
    framelink = FrameLink(Required=['x', 'z'], label='PF', delim='_')
    framelink.insert(2, range(2))
    assert framelink.index.to_list() == ['PF_0', 'PF_1']


def test_columns():
    framelink = FrameLink(metadata={'Required': ['x', 'z'],
                                    'Additional': ['rms']})
    framelink.insert(2, [5, 6, 7], rms=5)
    framelink = FrameLink(framelink, columns=['x', 'z'])
    assert framelink.columns.to_list() == ['x', 'z']


def test_columns_metaframe_update():
    framelink = FrameLink(metadata={'Required': ['x', 'z', 'dl'],
                                    'Additional': ['rms']},
                          columns=['x', 'dt'])
    framelink.insert(4, dt=[5, 7, 12])
    framelink = FrameLink(framelink, columns=['x', 'dt', 'dl'])
    assert framelink.columns.to_list() == ['x', 'dt', 'dl']


def test_index_length_error():
    framelink = FrameLink()
    with pytest.raises(IndexError):
        framelink.insert(4, [5, 4, 6], 0.1, 0.3, name=['coil0', 'coil1'])


def test_required_add_frame():
    framelink = FrameLink(metadata={'Required': ['x', 'z']})
    framelink.insert(1, 2)


def test_required_add_frame_error():
    framelink = FrameLink(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        framelink.insert(1, 2, 3)


def test_attribute_metadata_replace():
    framelink = FrameLink(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    framelink.insert(3, 4, metadata={'Required': ['x', 'z']})


def test_attribute_metadata_replace_error():
    framelink = FrameLink(metadata={'Required': ['x', 'z', 'dl', 'dt']})
    with pytest.raises(IndexError):
        framelink.insert(3, 4, metadata={'required': ['x', 'z']})


def test_required_additional_metadata_clash():
    framelink = FrameLink(Required=['x', 'z'], Additional=['x', 'rms'])
    framelink.insert(3, 4)
    assert list(framelink.columns) == ['x', 'z', 'rms']


def test_frame_columns():
    framelink = FrameLink(metadata={'Required': ['x', 'z'], 'Additional': []})
    framelink.insert(0, 1)
    assert list(framelink.columns) == ['x', 'z']


def test_Ic_unset():
    framelink = FrameLink(Required=['x'])
    framelink.insert([4, 5], It=6.5)
    assert framelink.Ic.to_list() == [6.5, 6.5]


def test_Ic_unset_Additional():
    framelink = FrameLink(Required=['x', 'z'], Additional=['Ic'])
    framelink.insert(4, range(2), It=5)
    assert framelink.Ic.to_list() == [5, 5]


def test_init_insert_label():
    framelink = FrameLink(Required=['x'], label='CS')
    framelink.insert(range(3))
    assert framelink.index.to_list() == ['CS0', 'CS1', 'CS2']


def test_insert_insert_label_switch():
    framelink = FrameLink(Required=['x', 'z'], label='CS')
    framelink.insert(4, 1, label='PF')
    framelink.insert(range(2), 1, offset=7)
    framelink.insert(2, range(2), label='PF')
    framelink.insert(range(2), 1)
    assert framelink.index.to_list() == \
        ['PF0', 'CS7', 'CS8', 'PF1', 'PF2', 'CS9', 'CS10']


def test_named_multi_insert():
    framelink = FrameLink(required=['x', 'z'], label='CS', offset=11)
    framelink.insert([-4, -5], 1, Ic=6.5, name='PF4')
    framelink.insert([-4, -5], 2, Ic=6.5)
    framelink.insert([-4, -5], 3, Ic=6.5, name='PF0')
    assert framelink.index.to_list() == \
        ['PF4', 'PF5', 'CS11', 'CS12', 'PF6', 'PF7']


def test_named_insert_no_number():
    framelink = FrameLink(required=['x', 'z'], label='CS', offset=11)
    framelink.insert([-4, -5], 1, Ic=6.5, name='PF')
    assert framelink.index.to_list() == ['PF11', 'PF12']


def test_named_insert_number_not_trailing():
    framelink = FrameLink(required=['x', 'z'], label='CS', offset=11)
    framelink.insert([-4, -5], 1, Ic=6.5, name='CS_3_PF')
    assert framelink.index.to_list() == ['CS_3_PF11', 'CS_3_PF12']


def test_labelled_insert_offset():
    framelink = FrameLink(required=['x', 'z'], label='CS', offset=7)
    framelink.insert([-4, -5], 1, Ic=6.5, label='PF')
    framelink.insert([-4, -5], 2, Ic=6.5)
    assert framelink.index.to_list() == ['PF7', 'PF8', 'CS7', 'CS8']


def test_required_insert():
    framelink = FrameLink(metadata={'Required': ['x', 'z']})
    framelink.insert(1, 2)


def test_required_add_error():
    framelink = FrameLink(metadata={'Required': ['x', 'z']})
    with pytest.raises(IndexError):
        assert framelink.insert(1, 2, 3)


def test_frame_index():
    framelink = FrameLink(metadata={'Required': ['x', 'z'], 'Additional': []})
    framelink.insert(0, 1)
    assert list(framelink.columns) == ['x', 'z']


def test_frame_addition():
    framelink = FrameLink(required=['x', 'y'])
    framelink += ([1, 2, 3], [4, 4, 4])
    framelink += dict(x=2, y=[4, 5])
    framelink += dict(x=3, y=3, z=22)
    assert framelink.x.to_list() == [1, 2, 3, 2, 2, 3]
    assert framelink.y.to_list() == [4, 4, 4, 4, 5, 3]
    assert framelink.z.to_list() == [0, 0, 0, 0, 0, 22]


def test_dataframe_addition():
    frame1 = FrameLink(required=['x', 'y'], label='a')
    frame2 = FrameLink(required=['x', 'y'], label='b')
    frame1 += dict(x=4, y=[1, 2, 3], z=1.2)
    frame2 += [5, 5, 5], [2, 3, 4]
    frame3 = frame1 + frame2
    assert frame3.index.to_list() == ['a0', 'a1', 'a2', 'b0', 'b1', 'b2']


def test_dataframe_addition_insert_required():
    frame1 = FrameLink(required=['x', 'y'], label='a')
    frame2 = FrameLink(required=['z'], label='b')
    frame1 += dict(x=4, y=[1, 2, 3], z=1.2)
    frame2 += [5, 5, 5],
    frame3 = frame1 + frame2
    assert frame3.z.to_list() == [1.2, 1.2, 1.2, 5, 5, 5]


def test_dataframe_inplace_addition_insert_required():
    frame1 = FrameLink(required=['x', 'y'], label='a')
    frame2 = FrameLink(required=['z'], label='b')
    frame1 += dict(x=4, y=[1, 2, 3], z=1.2)
    frame2 += [5, 5, 5],
    frame2 += frame1
    assert frame2.z.to_list() == [5, 5, 5, 1.2, 1.2, 1.2]


def test_frame_addition_required_error():
    framelink = FrameLink(required=['x', 'y'])
    with pytest.raises(IndexError):
        framelink += dict(x=1, z=2)


def test_frame_insert_required():
    framelink = FrameLink(required=['x', 'y'])
    with framelink.insert_required(['x']):
        framelink.insert([4, 5])
    assert framelink.x.to_list() == [4, 5]
    assert framelink.y.to_list() == [0, 0]


if __name__ == '__main__':

    pytest.main([__file__])
