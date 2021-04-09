
import pytest

from nova.electromagnetic.dataframe import (
    ColumnError, SubSpaceLockError, SubSpaceColumnError
    )
from nova.electromagnetic.frame import Frame


def test_init():
    frame = Frame(link=True, required=['x', 'z', 'dl', 'dt'],
                  metadata={'additional': ['link']})
    frame.insert(4, [5, 7, 12], 0.1, 0.05)
    return frame


def test_len():
    frame = Frame(Required=['x', 'z'])
    frame.insert(4, range(30), link=True)
    frame.insert(4, range(2), link=False)
    frame.insert(4, range(4), link=True)
    assert len(frame.subspace) == 4 and len(frame) == 36


def test_getattr():
    frame = Frame(Required=['x', 'z'])
    frame.insert(4, range(1), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    assert frame.Ic.to_list() == [5.0, 0.0, 0.0]


def test_getitem():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(3), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    assert frame.subspace['Ic'].to_list() == [5.0, 0.0, 0.0]


def test_setattr():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(7), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace.Ic = [3.6, 5.2, 10]
    assert frame.subspace.Ic.to_list() == [3.6, 5.2, 10.0]


def test_setattr_error():
    frame = Frame(Required=['x', 'z'], Subspace=[])
    frame.insert(4, range(7), Ic=5, nturn=3.6, link=True)
    with pytest.raises(ColumnError):
        frame.subspace.It = [3.6, 5.2, 10.4]


def test_setattr_current():
    frame = Frame(Required=['x', 'z'], Subspace=['It'])
    frame.insert(4, range(7), Ic=5, nturn=3.6, link=True)
    frame.insert(4, range(2), nturn=5.2, link=False)
    frame.subspace.It = [3.6, 5.2, 10.4]
    assert frame.subspace.Ic.to_list() == [1, 1, 2]


def test_setitem():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(5), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace['Ic'] = [3.6, 5.2, 10]
    assert frame.subspace.Ic.to_list() == [3.6, 5.2, 10.0]


def test_loc():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic', 'It'])
    frame.insert(4, range(2), Ic=5, link=True)
    frame.insert(4, range(2), Ic=0, link=False)
    frame.subspace.loc[:, 'It'] = [3.6, 5.2, 0]
    assert frame.subspace.It.to_list() == [3.6, 5.2, 0]


def test_loc_slice():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'],
                  label='Coil', offset=15)
    frame.insert(4, range(2), It=5, link=True)
    frame.insert(4, range(2), It=7.3, link=False)
    frame.loc['Coil15':'Coil17', 'It'] = [3.6, 3.6, 5.2]


def test_loc_error():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic', 'It'])
    frame.insert(4, range(2), Ic=5, link=True)
    frame.insert(4, range(2), Ic=0, link=False)
    with pytest.raises(SubSpaceLockError):
        frame.loc[:, 'It'] = [3.6, 5.2, 0]


def test_iloc():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(7), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace.Ic = 0
    frame.subspace.iloc[1, 0] = 3.6
    assert frame.subspace.Ic.to_list() == [0, 3.6, 0]


def test_set_at():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(7), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace.Ic = 0.0
    frame.subspace.at['Coil7', 'Ic'] = 3.6
    assert frame.subspace.Ic.to_list() == [0, 3.6, 0]


def test_set_iat():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(7), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace.Ic = 0.0
    frame.subspace.iat[-2, 0] = 3.6
    assert frame.subspace.Ic.to_list() == [0, 3.6, 0]


def test_get_at():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(7), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace.Ic = [7.4, 3.2, 6.666]
    assert frame.subspace.at['Coil8', 'Ic'] == 6.666


def test_get_at_keyerror():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(7), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(KeyError):
        _ = frame.subspace.at['Coil6', 'Ic']


def test_get_iat_indexerror():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(7), Ic=5, link=True)
    frame.insert(4, range(2), link=False)
    frame.subspace.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(IndexError):
        _ = frame.subspace.iat[7, 2]


def test_get_frame():
    frame = Frame(Required=['x', 'z'])
    frame.insert(4, range(3), Ic=5.7, link=True)
    frame.insert(4, range(2), Ic=3.2, link=False)
    with frame.setlock(True, 'subspace'):
        assert frame.Ic.to_list() == [5.7, 5.7, 5.7, 3.2, 3.2]


def test_setattr_value_error():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(4, range(3), Ic=5.7, link=True)
    with pytest.raises(ValueError):
        frame.subspace.Ic = range(3)


def test_setitem_value_error():
    frame = Frame(Required=['x', 'z'], Array=[], Subspace=['Ic'])
    frame.insert(4, range(3), Ic=5.7, link=True)
    with pytest.raises(ValueError):
        frame.subspace['Ic'] = range(3)


def test_subspace_lock():
    frame = Frame(metadata={'Required': ['x'], 'Subspace': ['x']})
    assert frame.metaframe.hascol('subspace', 'x')
    assert frame.lock('subspace') is False


def test_subarray():
    frame = Frame(Required=['Ic'], Array=['Ic'], Subspace=['Ic'])
    frame.insert([7.6, 5.5], link=True)
    frame.insert([3, 3], link=True)
    _ = frame.subspace.loc[:, 'Ic']


def test_frame():
    frame = Frame(Required=['Ic'], Array=['Ic'], Subspace=['Ic'])
    frame.insert([7.6, 5.5], link=True)
    frame.insert([3, 3], link=True)
    _ = frame.loc[:, 'Ic']


def test_link_lock():
    frame = Frame(Required=['Ic'], Array=['Ic'], Subspace=['Ic'])
    with frame.setlock(True, 'array'):
        assert frame.subspace.lock('array') is True


def test_loc_space_access():
    frame = Frame(Required=['x'], Array=[], Subspace=['x'])
    frame.insert([5.5, 72.4], link=True, Ic=5.7)
    frame.insert([5.5, 72.4], link=True)
    _ = frame.loc['Coil1', 'x']


def test_set_loc_subspace_lock_error():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SubSpaceLockError):
        frame.loc[:, 'Ic'] = 6.6


def test_set_iloc_subspace_lock_error():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SubSpaceLockError):
        frame.iloc[:, 2] = 6.6


def test_set_loc_label_subspace_lock_error():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SubSpaceLockError):
        frame.loc['Coil0', 'Ic'] = 6.6


def test_set_iloc_row_subspace_lock_error():
    frame = Frame(Required=['x', 'z'], Subspace=['Ic'])
    frame.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SubSpaceLockError):
        frame.iloc[0, 2] = 6.6


def test_set_loc_subspace_column_error():
    frame = Frame(Required=['x', 'z'], Subspace=[])
    frame.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(SubSpaceColumnError):
        frame.subspace.loc[:, 'Ic'] = 6.6


def test_set_iat_not_in_subspace_index_error():
    frame = Frame(Required=['x', 'z'], Subspace=[])
    frame.insert(0.5, [6, 8.3], nturn=0.5, link=True)
    with pytest.raises(IndexError):
        frame.subspace.iat[0, 2] = 6.6


if __name__ == '__main__':

    pytest.main([__file__])
