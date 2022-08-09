"""Manage assembly transforms."""
from dataclasses import dataclass

import numpy as np
import scipy.spatial.transform
import xarray


@dataclass
class Rotate:
    """Provide clocking transform."""

    ncoil: int = 18

    def rotate(self, reverse=False):
        """Return clocking rotation transform about z-axis."""
        half_angle = self.half_angle
        if reverse:
            half_angle *= -1
        return scipy.spatial.transform.Rotation.from_euler('z', half_angle)

    @property
    def half_angle(self):
        """Return coilset half-angle."""
        return np.pi / self.ncoil

    def clock(self, vector):
        """Return clocking transform."""
        return self.rotate().apply(vector)

    def anticlock(self, vector):
        """Return reveresed clocking transform."""
        return self.rotate(reverse=True).apply(vector)

    def to_cylindrical(self, dataarray: xarray.DataArray) -> xarray.DataArray:
        """Retun dataarray in cylindrical coordinates."""
        phi = np.arctan2(dataarray[..., 1], dataarray[..., 0])
        dataarray = dataarray.copy().rename(
            dict(space='cylindrical')).assign_coords(
                dict(cylindrical=['r', 'rphi', 'z']))
        dataarray[..., 0] = np.linalg.norm(dataarray[..., :2], axis=-1)
        dataarray[..., 1] = dataarray[..., 0] * phi.values
        return dataarray


if __name__ == '__main__':

    print(Rotate().anticlock(Rotate().clock([1, 2, 3])))
