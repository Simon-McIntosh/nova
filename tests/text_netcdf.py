import pytest

from nova.database.netcdf import netCDF


def test_netcdf_ext():
    netcdf = netCDF()
    netcdf.set_path()
    netCDF().get_filepath()


if __name__ == '__main__':

    pytest.main([__file__])
