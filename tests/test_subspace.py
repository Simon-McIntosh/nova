
import pytest
import numpy as np

from nova.frame.error import (
    ColumnError, SpaceKeyError, SubSpaceKeyError
    )
from nova.frame.framespace import FrameSpace


def test_init():
    framespace = FrameSpace(link=True, required=['x', 'z', 'dl', 'dt'],
                            metadata={'additional': ['link']})
    framespace.insert(4, [5, 7, 12], 0.1, 0.05)
    assert framespace.shape == (3, 8)


def test_len():
    framespace = FrameSpace(Required=['x', 'z'])
    framespace.insert(4, range(30), link=True)
    framespace.insert(4, range(2), link=False)
    framespace.insert(4, range(4), link=True)
    assert len(framespace.subspace) == 4 and len(framespace) == 36


def test_getattr():
    framespace = FrameSpace(Required=['x', 'z'])
    framespace.insert(4, range(1), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    assert framespace.Ic.to_list() == [5.0, 0.0, 0.0]


def test_getitem():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(3), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    assert framespace.subspace['Ic'].to_list() == [5.0, 0.0, 0.0]


def test_setattr():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(7), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace.Ic = [3.6, 5.2, 10]
    assert framespace.subspace.Ic.to_list() == [3.6, 5.2, 10.0]


def test_setattr_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=[])
    framespace.insert(4, range(7), Ic=5, nturn=3.6, link=True)
    with pytest.raises(ColumnError):
        framespace.subspace.It = [3.6, 5.2, 10.4]


def test_setattr_current():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['It'])
    framespace.insert(4, range(7), Ic=5, nturn=3.6, link=True)
    framespace.insert(4, range(2), nturn=5.2, link=False)
    framespace.subspace.It = [3.6, 5.2, 10.4]
    assert framespace.subspace.Ic.to_list() == [1, 1, 2]


def test_setitem():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(5), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace['Ic'] = [3.6, 5.2, 10]
    assert framespace.subspace.Ic.to_list() == [3.6, 5.2, 10.0]


def test_loc():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'])
    framespace.insert(4, range(2), Ic=5, link=True)
    framespace.insert(4, range(2), Ic=0, link=False)
    framespace.subspace.loc[:, 'It'] = [3.6, 5.2, 0]
    assert framespace.subspace.It.to_list() == [3.6, 5.2, 0]


def test_loc_slice():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'],
                            label='Coil', offset=15)
    framespace.insert(4, range(2), It=5, link=True)
    framespace.insert(4, range(2), It=7.3, link=False)
    framespace.loc['Coil15':'Coil17', 'It'] = [3.6, 3.6, 5.2]


def test_loc_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'])
    framespace.insert(4, range(2), Ic=5, link=True)
    framespace.insert(4, range(2), Ic=0, link=False)
    with pytest.raises(SpaceKeyError):
        framespace.loc[:, 'It'] = [3.6, 5.2, 0]


def test_iloc():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(7), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace.Ic = 0
    framespace.subspace.iloc[1, 0] = 3.6
    assert framespace.subspace.Ic.to_list() == [0, 3.6, 0]


def test_set_at():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(7), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace.Ic = 0.0
    framespace.subspace.at['Coil7', 'Ic'] = 3.6
    assert framespace.subspace.Ic.to_list() == [0, 3.6, 0]


def test_set_iat():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(7), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace.Ic = 0.0
    framespace.subspace.iat[-2, 0] = 3.6
    assert framespace.subspace.Ic.to_list() == [0, 3.6, 0]


def test_get_at():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(7), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace.Ic = [7.4, 3.2, 6.666]
    assert framespace.subspace.at['Coil8', 'Ic'] == 6.666


def test_get_at_keyerror():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(7), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(KeyError):
        _ = framespace.subspace.at['Coil6', 'Ic']


def test_get_iat_indexerror():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(7), Ic=5, link=True)
    framespace.insert(4, range(2), link=False)
    framespace.subspace.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(IndexError):
        _ = framespace.subspace.iat[7, 2]


def test_get_frame():
    framespace = FrameSpace(Required=['x', 'z'])
    framespace.insert(4, range(3), Ic=5.7, link=True)
    framespace.insert(4, range(2), Ic=3.2, link=False)
    with framespace.setlock(True, 'subspace'):
        assert framespace.Ic.to_list() == [5.7, 5.7, 5.7, 3.2, 3.2]


def test_setattr_value_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(4, range(3), Ic=5.7, link=True)
    with pytest.raises(ValueError):
        framespace.subspace.Ic = range(3)


def test_setitem_value_error():
    framespace = FrameSpace(Required=['x', 'z'], Array=[], Subspace=['Ic'])
    framespace.insert(4, range(3), Ic=5.7, link=True)
    with pytest.raises(ValueError):
        framespace.subspace['Ic'] = range(3)


def test_subspace_lock():
    framespace = FrameSpace(metadata={'Required': ['x'], 'Subspace': ['x']})
    assert framespace.metaframe.hascol('subspace', 'x')
    assert framespace.lock('subspace') is False


def test_subarray():
    framespace = FrameSpace(Required=['Ic'], Array=['Ic'], Subspace=['Ic'])
    framespace.insert([7.6, 5.5], link=True)
    framespace.insert([3, 3], link=True)
    _ = framespace.subspace.loc[:, 'Ic']


def test_frame():
    framespace = FrameSpace(Required=['Ic'], Array=['Ic'], Subspace=['Ic'])
    framespace.insert([7.6, 5.5], link=True)
    framespace.insert([3, 3], link=True)
    _ = framespace.loc[:, 'Ic']


def test_link_lock():
    framespace = FrameSpace(Required=['Ic'], Array=['Ic'], Subspace=['Ic'])
    with framespace.setlock(True, 'array'):
        assert framespace.subspace.lock('array') is True


def test_loc_space_access():
    framespace = FrameSpace(Required=['x'], Array=[], Subspace=['x'])
    framespace.insert([5.5, 72.4], link=True, Ic=5.7)
    framespace.insert([5.5, 72.4], link=True)
    _ = framespace.loc['Coil1', 'x']


def test_set_loc_subspace_lock_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SpaceKeyError):
        framespace.loc[:, 'Ic'] = 6.6


def test_set_iloc_subspace_lock_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SpaceKeyError):
        framespace.iloc[:, 2] = 6.6


def test_set_loc_label_subspace_lock_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SpaceKeyError):
        framespace.loc['Coil0', 'Ic'] = 6.6


def test_set_iloc_row_subspace_lock_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SpaceKeyError):
        framespace.iloc[0, 2] = 6.6


def test_set_loc_subspace_column_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=[])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SubSpaceKeyError):
        framespace.subspace.loc[:, 'Ic'] = 6.6


def test_set_iat_not_in_subspace_index_error():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=[])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(IndexError):
        framespace.subspace.iat[0, 2] = 6.6


def test_insert_negative_factor_Ic():
    framespace = FrameSpace({'x': [1, 3], 'z': 0}, factor=0.5, name='Coil0',
                            link=True,
                            metadata={'Required': ['x', 'z']},
                            subspace=['Ic'])
    framespace.insert(4, [7, 8], link=True, factor=-0.5)
    framespace.insert(4, [7, 8], link=True, factor=-1)
    framespace.subspace.Ic = 10.
    assert framespace.loc[:, 'Ic'].to_list() == [10, 5, 10, -5, 10, -10]


def test_set_loc_subspace_Ic():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    framespace.subspace.loc[:, 'Ic'] = 6.6
    assert framespace.loc[:, 'It'].to_list() == [3.3, 3.3]


def test_set_loc_subspace_Ic_It():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    framespace.subspace.loc[:, 'Ic'] = 6.6
    assert framespace.loc[:, 'It'].to_list() == [3.3, 3.3]


def test_set_loc_It():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'])
    framespace.insert(0.5, [6, 8.3], nturn=0.25)
    framespace.subspace.loc[:, 'It'] = 6.6
    assert framespace.subspace.loc[:, 'Ic'].to_list() == [26.4, 26.4]


def test_set_subspace_Ic_It_repr():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'])
    framespace.insert(0.5, [6, 8.3], nturn=0.25)
    framespace.__repr__()


def test_subspace_Ic():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5)
    framespace.insert(0.5, range(10), nturn=3.5, link=True)
    framespace.subspace.Ic = [6.6, 6.6, 1]
    assert framespace.subspace.It.to_list() == [3.3, 3.3, 3.5]


def test_subspace_intersect_columns():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5)
    framespace.insert(0.5, range(10), nturn=3.5, link=True)
    framespace.subspace.Ic = [6.6, 6.6, 1]
    framespace.update_frame()


def test_set_It_array():
    framespace = FrameSpace(Required=['x', 'z'], Subspace=['Ic', 'It'],
                  Array=['Ic', 'nturn', 'It'])
    framespace.insert(0.5, [6, 8.3], nturn=0.5)
    framespace.insert(0.5, range(10), nturn=4, link=True)
    framespace.subspace.It = [6.6, 6.6, 1]
    assert np.isclose(framespace.subspace.Ic, [13.2, 13.2, 0.25]).all()




if __name__ == '__main__':

    pytest.main([__file__])
