
import pytest
import numpy as np
import pandas

from nova.electromagnetic.dataframearray import DataFrameArray
from nova.electromagnetic.metaarray import MetaArray


def test_init_update():
    frame = DataFrameArray({'x': [3, 2], 'z': 0},
                           Required=['x', 'z'], Array=['x', 'z'])
    frame.x = [1, 2]
    frame = DataFrameArray(frame)
    print(frame)
    print(frame.metaarray)
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert list(update_array.values()) == [False, True]
    assert list(update_frame.values()) == [False, True]


def test_exclude_internal_metadata():
    metaarray = MetaArray()
    metaarray._internal = ['data']
    assert 'data' not in metaarray.metadata and hasattr(metaarray, 'data')


def test_array_ndarray():
    frame = DataFrameArray({'x': [3, 2], 'z': 0},
                           Required=['x', 'z'], Array=['x'])
    assert isinstance(frame.x, np.ndarray)


def test_get_array_update():
    frame = DataFrameArray({'x': [3, 2], 'z': 0},
                           Required=['x', 'z'], Array=['x', 'z'])
    _ = frame.x
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert not update_array['x'] and not update_frame['x']


def test_set_array_update():
    frame = DataFrameArray({'x': [3, 2], 'z': 0},
                           Required=['x', 'z'], Array=['x'])
    frame.x = [1, 2]
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert not update_array['x'] and update_frame['x']


def test_reload_frame():
    frame = DataFrameArray({'x': [3, 2], 'z': 0}, Array=['x', 'z'])
    frame.x = [1, 7]
    frame.z = [4, 5]
    assert frame['x'].to_list() == [1, 7] and frame['z'].to_list() == [4, 5]


def test_update_array():
    frame = DataFrameArray({'x': [3, 2, 5], 'z': 0}, Array=['x'])
    frame.x = [1, 2, 3]
    assert list(frame.x) == [1, 2, 3]


def test_set_slice_on_array_variable():
    frame = DataFrameArray({'x': [3, 2, 5], 'z': 0}, Array=['x'])
    frame.x = [1, 2, 3]
    frame.x[1:] = [6, 7]
    assert list(frame.x) == [1, 6, 7]


def test_warn_set_slice_on_frame():
    frame = DataFrameArray({'x': [3, 2, 5, 7, 6], 'z': 0}, Array=['x'])
    index = frame.x < 4
    with pytest.warns(pandas.core.common.SettingWithCopyWarning):
        frame.loc[index].x = 5


def test_set_slice_on_col():
    frame = DataFrameArray({'x': [3, 2, 5, 7, 6], 'z': 0},
                           Array=[])
    frame['x'][-2:] = [6, 7]
    assert frame.x.to_list()[-2:] == [6, 7]


if __name__ == '__main__':

    pytest.main([__file__])
