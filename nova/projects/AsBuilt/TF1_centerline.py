
from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import xarray

from nova.definitions import root_dir


@dataclass
class CenterLine:

    file: str = 'TFC1_CL'
    directory: str = None
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        if self.directory is None:
            self.directory = os.path.join(root_dir, 'input/geometry/ITER')

        self.data = xarray.Dataset(
            coords=dict(loadcase=['warm', 'cool', 'TFonly'],
                        coil=[f'TF{i+1}' for i in range(18)],
                        position=['x', 'y', 'z']))

        filepath = os.path.join(self.directory, f'{self.file}.xlsx')
        with pandas.ExcelFile(filepath, engine='openpyxl') as xls:
            for i, sheet_name in enumerate(xls.sheet_names):
                name = f'dp{i}'
                index = f'index_{i}'
                double_pancake = self.read_sheet(xls, sheet_name)
                # initialize
                self.data[name] = xarray.DataArray(
                    np.zeros((len(double_pancake), 3, 18, 3)),
                    dims=[index, 'position', 'coil', 'loadcase'],
                    coords={index: double_pancake.index})
                # populate TF1
                self.data[name].loc[:, :, 'TF1', 'warm'] = \
                    double_pancake.to_numpy()

    def read_sheet(self, xls, sheet_name):
        """Read excel worksheet."""
        sheet = pandas.read_excel(xls, sheet_name, usecols=[2, 3, 4])
        columns = {'X Coord': 'x', 'Y Coord': 'y', 'Z Coord': 'z'}
        sheet.rename(columns=columns, inplace=True)
        return sheet



if __name__ == '__main__':

    cl = CenterLine()
