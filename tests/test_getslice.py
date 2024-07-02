import pytest

import numpy as np
import xarray

from nova.imas.getslice import GetSlice

DATA = {
    "elongation": ("time", [2.1, 2.2, 1.8]),
    "minor_radius": ("time", [0.1, 1.2, 3.2]),
    "dcoil": -1,
    "time": [1, 2, 3.3],
}


@pytest.fixture()
def data():
    return xarray.Dataset(DATA).copy(deep=True)


def test_match():
    getslice = GetSlice()
    assert getslice.match("p_prime") == "dpressure_dpsi"
    assert getslice.match("ff_prime") == "f_df_dpsi"
    assert getslice.match("another") == "another"
    with pytest.raises(TypeError):
        getslice.match(5)


def test_xarray_copy():
    data = xarray.Dataset({"a": [1, 2, 3]})
    getslice = GetSlice(data=data)
    datacopy = getslice.copy(getslice.data)
    assert np.allclose(datacopy.a, data.a)
    data["a"] = [3, 4, 5]
    assert not np.allclose(datacopy.a, data.a)


def test_dict_copy():
    attr = {"a": [1, 2, 3]}
    getslice = GetSlice()
    getslice.attr = attr
    attrcopy = getslice.copy(getslice.attr)
    assert np.allclose(attrcopy["a"], attr["a"])
    attr["a"] = [3, 4, 5]
    assert not np.allclose(attrcopy["a"], attr["a"])


def test_float_copy():
    attr = 7.3
    getslice = GetSlice()
    getslice.attr = attr
    attrcopy = getslice.copy(getslice.attr)
    assert np.isclose(attrcopy, attr)
    attr = 6.3
    assert not np.isclose(attrcopy, attr)


@pytest.mark.parametrize("time_index", [1, -1])
def test_get_data(data, time_index):
    getslice = GetSlice(data)
    getslice.itime = time_index
    assert getslice.time_index == time_index
    assert getslice.itime == time_index
    assert np.isclose(
        getslice.get_data("elongation"), DATA["elongation"][1][time_index]
    )
    assert getslice._cache == {
        attr: getslice._cache.get(attr, None) for attr in getslice.persist
    }


@pytest.mark.parametrize("time_index", [1, 0])
def test_getitime(data, time_index):
    getslice = GetSlice(data)
    getslice.itime = time_index
    assert np.isclose(getslice["minor_radius"], DATA["minor_radius"][1][time_index])
    assert "minor_radius" in getslice._cache
    getslice.update()
    assert "minor_radius" not in getslice._cache


def test_time_partition(data):
    getslice = GetSlice(data)
    _partition = DATA["time"].copy()
    _partition[:-1] += np.diff(DATA["time"]) / 2
    assert np.allclose(getslice._partition, _partition)


def test_get_itime(data):
    getslice = GetSlice(data)
    getslice.time = DATA["time"][0] + 0.01
    assert getslice.itime == 0
    getslice.time = getslice._partition[0]
    assert getslice.itime == 0
    getslice.time = getslice._partition[0] + 0.1
    assert getslice.itime == 1
    getslice.time = DATA["time"][1]
    assert getslice.itime == 1
    getslice.time = DATA["time"][2]
    assert getslice.itime == 2


@pytest.mark.parametrize("time_index", range(3))
def test_time(data, time_index):
    getslice = GetSlice(data)
    getslice.itime = time_index
    assert np.isclose(getslice.time, DATA["time"][time_index])


def test_cache_reset(data):
    getslice = GetSlice(data.copy(deep=True))
    with pytest.raises(IndexError):
        getslice["minor_radius"] = 5.2
    getslice.itime = 2
    getslice["minor_radius"] = 5.2
    assert not np.allclose(DATA["minor_radius"][1], getslice.data.minor_radius)
    getslice.reset()
    assert getslice._cache["data"] is None
    getslice.data = data
    getslice.cache()
    assert np.allclose(DATA["minor_radius"][1], getslice.data.minor_radius)


if __name__ == "__main__":
    pytest.main([__file__])
