
import pytest
import numpy as np
import pandas

from nova.electromagnetic.frame import Frame


def test_generate_key_attribute_true():
    frame = Frame({'link': [True]}, Required=[])
    print(frame.metaframe)
    assert frame.multipoint.generate
test_generate_key_attribute_true()


def test_generate_single():
    frame = Frame({'link': [True]}, metadata={'Required': []})
    assert frame.iloc[0].to_list() == ['', 1.0, 0, 0]


def test_generate_single_float():
    frame = Frame({'link': [-0.3]}, metadata={'Required': []})
    assert frame.iloc[0].to_list() == ['', 1.0, 0, 0]


def test_generate_multi():
    frame = Frame({'x': range(3), 'link': True}, name='coil1',
                  metadata={'Required': ['x']})
    assert frame.link.to_list() == ['', 'coil1', 'coil1']


def test_data_init_multipoint_true():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'link': True})
    frame = Frame(data, name='Coil0', metadata={'Required': ['x', 'z']})
    assert frame.link.to_list() == ['', 'Coil0', 'Coil0'] and \
        frame.factor.to_list() == [1, 1, 1]


def test_add_coil_multipoint_default_false():
    frame = Frame({'x': [1, 3], 'z': 0}, link=False, name='Coil0',
                  metadata={'Required': ['x', 'z'], 'Additional': []})
    frame.insert(4, [7, 8], link=True)
    assert frame.link.to_list() == ['', '', '', 'Coil2'] and \
        frame.factor.to_list() == [1, 1, 1, 1]


def test_add_coil_multipoint_default_true():
    frame = Frame({'x': [1, 3], 'z': 0}, link=True, name='Coil0',
                  metadata={'Required': ['x', 'z'], 'Additional': []})
    frame.insert(4, [7, 8], link=True)
    assert frame.link.to_list() == ['', 'Coil0', '', 'Coil2'] and \
        frame.factor.to_list() == [1, 1, 1, 1]


def test_add_coil_multipoint_default_float():
    frame = Frame({'x': [1, 3], 'z': 0}, link=0.5, name='Coil0',
                  metadata={'Required': ['x', 'z'], 'Additional': []})
    frame.insert(4, [7, 8], link=True)
    assert frame.link.to_list() == ['', 'Coil0', '', 'Coil2'] and \
        frame.factor.to_list() == [1, 0.5, 1, 1]


def test_default_multipoint_true():
    frame = Frame(link=True, metadata={'Required': ['x', 'z'],
                                       'Additional': ['link']})
    frame.insert(4, [5, 7, 12], name='coil1')
    assert frame.link.to_list() == ['', 'coil1', 'coil1'] and \
        frame.factor.to_list() == [1, 1, 1]


def test_data_init_multipoint_false():
    data = pandas.DataFrame({'x': 3, 'z': [3, 6, 8], 'link': False})
    frame = Frame(data, name='Coil0', metadata={'Required': ['x', 'z'],
                                                'Additional': ['link']})
    unset = [link == '' for link in frame.link]
    assert np.array(unset).all()


def test_drop():
    frame = Frame(link=True, label='coil',
                  metadata={'Required': ['x', 'z'], 'Additional': ['link']})
    frame.insert(4, [5, 7, 12])
    frame.insert(3, [1, 2], link=False)
    frame.insert(6, [7, 3])
    frame.insert(12, [7, 3])
    frame.multipoint.drop(['coil0', 'coil7'])
    assert frame.link.to_list() == ['', '', '', '', '', '', 'coil5', '', '']
    assert frame.factor.to_list() == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert frame.ref.to_list() == [0, 1, 2, 3, 4, 5, 5, 7, 8]
    assert frame.subref.to_list() == [0, 1, 2, 3, 4, 5, 5, 6, 7]


def test_drop_indexer():
    frame = Frame(link=True, metadata={'Required': ['x', 'z'],
                                       'Additional': ['link']})
    frame.insert(4, [5, 7, 12], label='coil')
    frame.insert(3, [1, 2])
    frame.multipoint.drop('coil0')
    assert frame.multipoint.indexer == [0, 1, 2, 3]


def test_frame_columns_multipoint():
    frame = Frame(metadata={'Required': ['x', 'z'], 'Additional': []})
    frame.insert(0, 1, link=True)
    assert list(frame.columns) == ['x', 'z', 'link', 'factor', 'ref', 'subref']


def test_link_avalible():
    frame = Frame(Required=['x'], Available=['link'])
    frame.insert(1)


if __name__ == '__main__':

    pytest.main([__file__])
