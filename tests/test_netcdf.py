import numpy as np
import pytest
import xarray

from nova.database.netcdf import netCDF


def test_filepath_suffix(tmp_path):
    netcdf = netCDF(filename="sample", dirname=str(tmp_path))
    assert netcdf.filepath.suffix == ".nc"
    assert netcdf.filepath.name == "sample.nc"


def test_store_load_roundtrip(tmp_path):
    data = xarray.Dataset({"current": ("coil", np.arange(4.0))})
    netCDF(filename="coils", dirname=str(tmp_path), group="pf", data=data).store()

    reloaded = netCDF(filename="coils", dirname=str(tmp_path), group="pf").load()
    np.testing.assert_array_equal(reloaded.data["current"].values, np.arange(4.0))


def test_subgroup():
    netcdf = netCDF(filename="sample", group="pf")
    assert netcdf.subgroup("active") == "pf/active"
    assert netCDF(filename="sample").subgroup() is None


def test_store_overwrite_replaces_group(tmp_path):
    netCDF(
        filename="coils",
        dirname=str(tmp_path),
        group="pf",
        data=xarray.Dataset({"current": ("coil", np.arange(3.0))}),
    ).store()
    netCDF(
        filename="coils",
        dirname=str(tmp_path),
        group="pf",
        data=xarray.Dataset({"current": ("coil", np.ones(3))}),
    ).store_overwrite()

    reloaded = netCDF(filename="coils", dirname=str(tmp_path), group="pf").load()
    np.testing.assert_array_equal(reloaded.data["current"].values, np.ones(3))


if __name__ == "__main__":
    pytest.main([__file__])
