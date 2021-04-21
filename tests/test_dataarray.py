
import pytest
import numpy as np
import pandas

from nova.electromagnetic.dataarray import DataArray
from nova.electromagnetic.metaframe import MetaFrame


def test_exclude_internal_metadata():
    metaframe = MetaFrame()
    metaframe._internal = ['data']
    assert 'data' not in metaframe.metadata and hasattr(metaframe, 'data')


def test_array_ndarray():
    dataarray = DataArray({'x': [3, 2], 'z': 0},
                          Required=['x', 'z'], Array=['x'])
    assert isinstance(dataarray.x, np.ndarray)


def test_array_getitem():
    dataarray = DataArray({'x': [3, 2], 'z': 0}, Array=['x', 'z'])
    dataarray.x = [1, 7]
    dataarray.z = [4, 5]
    assert list(dataarray['x']) == [1, 7]
    assert list(dataarray['z']) == [4, 5]


def test_update_array():
    dataarray = DataArray({'x': [3, 2, 5], 'z': 0}, Array=['x'])
    dataarray.x = [1, 2, 3]
    assert list(dataarray.x) == [1, 2, 3]


def test_set_slice_on_array_variable():
    dataarray = DataArray({'x': [3, 2, 5], 'z': 0}, Array=['x'])
    dataarray.x = [1, 2, 3]
    dataarray.x[1:] = [6, 7]
    assert list(dataarray.x) == [1, 6, 7]


def test_warn_set_slice_on_frame():
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=['x'])
    index = dataarray.x < 4
    with pytest.warns(pandas.core.common.SettingWithCopyWarning):
        dataarray.loc[index].x = 5


def test_set_slice_on_col():
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=[])
    dataarray['x'][-2:] = [6, 7]
    assert dataarray.x.to_list()[-2:] == [6, 7]


def test_getattr_numpy():
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=['x'])
    assert isinstance(dataarray.x, np.ndarray)


def test_getattr_slice():
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=['x'])
    dataarray.x = range(5)
    assert np.isclose(dataarray.x[:3], np.array([0, 1, 2])).all()


def test_setattr_slice():
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=['x'])
    dataarray.x[:3] = range(3)
    assert np.isclose(dataarray.x, np.array([0, 1, 2, 7, 6])).all()


def test_setitem_slice():
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=['x'])
    dataarray['x'][:3] = range(3)
    assert np.isclose(dataarray.x, np.array([0, 1, 2, 7, 6])).all()


def test_setitem_slice_warn():
    dataarray = DataArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=[])
    dataarray['x'][:3] = range(3)
    assert np.isclose(dataarray.x, np.array([0, 1, 2, 7, 6])).all()


def test_loc_update_single():
    dataarray = DataArray({'x': range(7), 'z': 0},
                          additional=['Ic'], Array=['Ic'], label='Coil')
    dataarray.Ic = 9
    dataarray.loc['Coil0', 'Ic'] = 7
    assert list(dataarray.Ic[:3]) == [7, 9, 9]


def test_loc_update_slice():
    dataarray = DataArray({'x': range(7), 'z': 0},
                          additional=['Ic'], Array=['Ic'], label='Coil')
    dataarray.Ic = 9
    dataarray.loc['Coil0':'Coil1', 'Ic'] = 7
    assert list(dataarray.Ic[:3]) == [7, 7, 9]


def test_get_loc_slice():
    dataarray = DataArray({'x': range(6)}, Required=['x'], Array=['x'])
    dataarray.loc['Coil0':'Coil1', 'x']
    assert dataarray.loc['Coil0':'Coil3', 'x'].to_list() == [0, 1, 2, 3]


def test_get_iloc_slice():
    dataarray = DataArray({'x': range(6)}, Required=['x'], Array=['x'])
    assert dataarray.iloc[0:4, 0].to_list() == [0, 1, 2, 3]


def test_set_get_loc_slice():
    dataarray = DataArray({'x': range(6)}, Required=['x'], Array=['x'])
    dataarray.x[:2] = 7.7
    assert dataarray.loc['Coil0':'Coil3', 'x'].to_list() == [7.7, 7.7, 2, 3]


def test_set_loc():
    dataarray = DataArray({'x': range(3)}, Required=['x'], Array=['x'])
    dataarray.loc[:, 'x'] = 7.7
    assert list(dataarray.x) == [7.7, 7.7, 7.7]


def test_set_loc_slice():
    dataarray = DataArray({'x': range(3)}, Required=['x'], Array=['x'])
    dataarray.loc['Coil0':'Coil1', 'x'] = 7.7
    assert list(dataarray.x) == [7.7, 7.7, 2.0]


def test_set_iloc():
    dataarray = DataArray({'x': range(3)}, Required=['x'], Array=['x'])
    dataarray.iloc[:, 'x'] = 7.7
    assert list(dataarray.x) == [7.7, 7.7, 7.7]


def test_set_iloc_slice():
    dataarray = DataArray({'x': range(3)}, Required=['x'], Array=['x'])
    dataarray.iloc['Coil0':'Coil1', 'x'] = 7.7
    assert list(dataarray.x) == [7.7, 7.7, 2.0]


def test_set_at():
    dataarray = DataArray({'x': range(3)}, Required=['x'], Array=['x'])
    dataarray.at['Coil0', 'x'] = 7.7
    assert list(dataarray.x) == [7.7, 1, 2]


def test_set_iat():
    dataarray = DataArray({'x': range(3)}, Required=['x'], Array=['x'])
    dataarray.iat[0, 0] = 7.7
    assert list(dataarray.x) == [7.7, 1, 2]


if __name__ == '__main__':

    pytest.main([__file__])
