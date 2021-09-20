"""Calculate placement error from fiducial data."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import xarray

from nova.structural.fiducialdata import FiducialData


@dataclass
class FiducialError:
    """Manage fiducial error estimates."""

    data: xarray.Dataset
    centerline: pv.PolyData = field(init=False)

    def calculate_midplane_error(self):
        """Calculate midplane radial and toroidal placement error."""
        self.centerline = pv.PolyData(1e-3*self.data.centerline[0].values)

        #gap = 1e3*(radius*dphi - np.mean(radius)*np.pi/9)

        #radius =

    def fit_to_line(self, index):
        """Fit coil to centerline."""


if __name__ == '__main__':

    data = FiducialData(fill=True, sead=2025).data
    error = FiducialError(data)
