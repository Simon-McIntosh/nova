"""Manage grid attributes."""
from dataclasses import dataclass

from nova.database.netcdf import netCDF


@dataclass
class FrameSetData(netCDF):
    """Frameset data baseclass."""

    group: str = 'framesetdata'

    def store(self, filename: str, path=None):
        """Store data as netCDF in hdf5 file."""
        print('****store plasma', self.file('tmp'))
        super().store(filename, path)
