
import pytest
import numpy as np
import pandas

from nova.frame.framelink import FrameLink


def test_generate_key_attribute_true():
    framelink = FrameLink({'link': [True]}, Required=[])
    assert framelink.multipoint.generate


def test_generate_single():
    framelink = FrameLink({'link': [True]}, metadata={'Required': []})
    assert framelink.iloc[0].to_list() == ['', 1.0, 0, 0]


def test_generate_single_float():
    framelink = FrameLink({'link': [-0.3]}, metadata={'Required': []})
    assert framelink.iloc[0].to_list() == ['', 1.0, 0, 0]


def test_generate_multi():
    framelink = FrameLink({'x': range(3), 'link': True}, name='coil1',
                          metadata={'Required': ['x']})
    assert framelink.link.to_list() == ['', 'coil1', 'coil1']


def test_data_init_multipoint_true():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'link': True})
    framelink = FrameLink(data, name='Coil0',
                          metadata={'Required': ['x', 'z']})
    assert framelink.link.to_list() == ['', 'Coil0', 'Coil0'] and \
        framelink.factor.to_list() == [1, 1, 1]


def test_insert_multipoint_default_false():
    framelink = FrameLink({'x': [1, 3], 'z': 0}, link=False, name='Coil0',
                          metadata={'Required': ['x', 'z'],
                                    'Additional': []})
    framelink.insert(4, [7, 8], link=True)
    assert framelink.link.to_list() == ['', '', '', 'Coil2'] and \
        framelink.factor.to_list() == [1, 1, 1, 1]


def test_insert_multipoint_default_true():
    framelink = FrameLink({'x': [1, 3], 'z': 0}, link=True, name='Coil0',
                          metadata={'Required': ['x', 'z'],
                                    'Additional': []})
    framelink.insert(4, [7, 8], link=True)
    assert framelink.link.to_list() == ['', 'Coil0', '', 'Coil2'] and \
        framelink.factor.to_list() == [1, 1, 1, 1]


def test_insert_multipoint_default_float():
    framelink = FrameLink({'x': [1, 3], 'z': 0}, factor=0.5, name='Coil0',
                          link=True, metadata={'Required': ['x', 'z'],
                                               'Additional': []})
    framelink.insert(4, [7, 8], link=True, factor=1)
    assert framelink.link.to_list() == ['', 'Coil0', '', 'Coil2'] and \
        framelink.factor.to_list() == [1, 0.5, 1, 1]


def test_default_multipoint_true():
    framelink = FrameLink(link=True, metadata={'Required': ['x', 'z'],
                                               'Additional': ['link']})
    framelink.insert(4, [5, 7, 12], name='coil1')
    assert framelink.link.to_list() == ['', 'coil1', 'coil1'] and \
        framelink.factor.to_list() == [1, 1, 1]


def test_data_init_multipoint_false():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'link': False})
    framelink = FrameLink(data, name='Coil0',
                          metadata={'Required': ['x', 'z'],
                                    'Additional': ['link']})
    unset = [link == '' for link in framelink.link]
    assert np.array(unset).all()


def test_drop():
    framelink = FrameLink(link=True, label='coil',
                          metadata={'Required': ['x', 'z'],
                                    'Additional': ['link']})
    framelink.insert(4, [5, 7, 12])
    framelink.insert(3, [1, 2], link=False)
    framelink.insert(6, [7, 3])
    framelink.insert(12, [7, 3])
    framelink.multipoint.drop(['coil0', 'coil7'])
    assert framelink.link.to_list() == ['', '', '', '', '', '',
                                        'coil5', '', '']
    assert framelink.factor.to_list() == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert framelink.ref.to_list() == [0, 1, 2, 3, 4, 5, 5, 7, 8]
    assert framelink.subref.to_list() == [0, 1, 2, 3, 4, 5, 5, 6, 7]


def test_drop_indexer():
    framelink = FrameLink(link=True, metadata={'Required': ['x', 'z'],
                                               'Additional': ['link']})
    framelink.insert(4, [5, 7, 12], label='coil')
    framelink.insert(3, [1, 2])
    framelink.multipoint.drop('coil0')
    assert framelink.multipoint.indexer == [0, 1, 2, 3]


def test_frame_columns_multipoint():
    framelink = FrameLink(metadata={'Required': ['x', 'z'],
                                    'Additional': []})
    framelink.insert(0, 1, link=True)
    assert list(framelink.columns) == ['x', 'z', 'link', 'factor',
                                       'ref', 'subref']


def test_link_avalible():
    framelink = FrameLink(Required=['x'], Available=['link'])
    framelink.insert(1)


def test_link_sort_normal():
    framelink = FrameLink(Required=['x'], Available=['link'], label='Coil')
    framelink.insert(range(4), link=['Coil0', '', 'Coil0', ''])
    assert framelink.link.to_list() == ['Coil0', '', 'Coil0', '']


def test_link_sort_reverse():
    framelink = FrameLink(Required=['x'], Available=['link'], label='Coil')
    framelink.insert(range(4), link=['Coil3', '', 'Coil3', ''])
    assert framelink.link.to_list() == ['', '', 'Coil0', 'Coil0']


if __name__ == '__main__':

    pytest.main([__file__])
