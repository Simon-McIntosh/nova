"""Manage IO as-built data."""
from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import pyvista as pv
from scipy.spatial.transform import Rotation
import xarray

from nova.definitions import root_dir
from nova.structural.datadir import DataDir
from nova.structural.fiducialdata import FiducialData
from nova.structural.fiducialerror import FiducialError
from nova.structural.plotter import Plotter


@dataclass
class AsBuilt:

    file: str = 'TFC18_asbuilt'
    ccl: pandas.DataFrame = field(init=False)

    def __post_init__(self):
        self.read_ccl()

    @property
    def xls_file(self):
        """Return xls filename."""
        return os.path.join(root_dir, 'input/geometry/ITER',
                            f'{self.file}.xlsx')

    @property
    def ccl_steps(self):
        """Return list of CCL assembly steps."""
        return self.data.columns.unique(0)[1:]

    def read_ccl(self):
        """Read xls file."""
        self.ccl = pandas.read_excel(self.xls_file, sheet_name='CCL',
                                     header=[0, 1], index_col=[0, 1])
        for step in self.ccl_steps:
            for coord in 'xyz':
                self.ccl.loc[:, (step, f'd{coord}')] = \
                    self.ccl.loc[:, (step, coord)] - \
                    self.ccl.loc[:, ('Nominal', coord)]

    def ccl_deltas(self):
        """Return dict of CCL deltas."""




if __name__ == '__main__':

    asbuilt = AsBuilt()
