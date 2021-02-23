
import pytest
import numpy as np
import pandas

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.array import MetaArray


def test_init_update():
    frame = Frame({'x': [3, 2], 'z': 0},
                  metadata={'Required': ['x', 'z'], 'Array': ['x', 'z']})
    frame.x = [1, 2]
    frame = Frame(frame)
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert np.array(list(update_array.values())).all() and \
        not np.array(list(update_frame.values())).any()


def test_exclude_internal_metadata():
    metaarray = MetaArray()
    metaarray._internal = ['data']
    assert 'data' not in metaarray.metadata and hasattr(metaarray, 'data')


def test_array_ndarray():
    frame = Frame({'x': [3, 2], 'z': 0}, metadata={'Required': ['x', 'z'],
                                                   'Array': ['x']})
    assert isinstance(frame.x, np.ndarray)


def test_get_array_update():
    frame = Frame({'x': [3, 2], 'z': 0}, metadata={'Required': ['x', 'z'],
                                                   'Array': ['x', 'z']})
    _ = frame.x
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert not update_array['x'] and not update_frame['x']


def test_set_array_update():
    frame = Frame({'x': [3, 2], 'z': 0}, metadata={'Required': ['x', 'z'],
                                                   'Array': ['x']})
    frame.x = [1, 2]
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert not update_array['x'] and update_frame['x']


def test_reload_frame():
    frame = Frame({'x': [3, 2], 'z': 0},
                  metadata={'Required': ['x', 'z'], 'Array': ['x', 'z']})
    frame.x = [1, 7]
    frame.z = [4, 5]
    assert frame['x'].to_list() == [1, 7] and frame['z'].to_list() == [4, 5]


def test_update_array():
    frame = Frame(metadata={'Array': ['x']})
    frame.add_frame(4, [5, 7, 12], 0.1, 0.05, section='r')
    frame.x = [1, 2, 3]
    assert list(frame.x) == [1, 2, 3]


def test_set_slice_on_array_variable():
    frame = Frame(metadata={'Array': ['x']})
    frame.add_frame(4, [5, 7, 12], 0.1, 0.05, section='r')
    frame.x = [1, 2, 3]
    frame.x[1:] = [6, 7]
    assert list(frame.x) == [1, 6, 7]


def test_warn_set_slice_on_nonarray_variable():
    frame = Frame(metadata={'Array': []})
    frame.add_frame(4, [5, 7, 12], 0.1, 0.05, section='r')
    with pytest.warns(pandas.core.common.SettingWithCopyWarning):
        frame.x[1:] = [6, 7]


def test_warn_set_slice_on_frame():
    frame = Frame(metadata={'Array': ['x']})
    frame.add_frame(4, [5, 7, 12], 0.1, 0.05, section='r')
    with pytest.warns(pandas.core.common.SettingWithCopyWarning):
        frame['x'][1:] = [6, 7]


if __name__ == '__main__':

    pytest.main([__file__])
