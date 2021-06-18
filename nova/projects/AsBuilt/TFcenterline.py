
from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import pyvista as pv
import xarray

from nova.definitions import root_dir


@dataclass
class TFCenterLine:
    """Load TFC1 referance winding pack centerline from file."""

    file: str = 'TFC1_CL'
    directory: str = None
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Load TF1 centerline."""
        if self.directory is None:
            self.directory = os.path.join(root_dir, 'input/geometry/ITER')
        self.load_single()

    def filepath(self, ext):
        """Return file path with ext extension."""
        return os.path.join(self.directory, f'{self.file}.{ext}')

    @property
    def ncdf_file(self):
        """Return netCDF filepath."""
        return self.filepath('nc')

    @property
    def xlsx_file(self):
        """Return xls filepath."""
        return self.filepath('xlsx')

    @property
    def vtm_file(self):
        """Return vtm filepath."""
        return self.filepath('vtm')

    def load_single(self):
        """Load centerline data."""
        try:
            self.data = xarray.open_dataset(self.ncdf_file)
        except FileNotFoundError:
            self.read_excel()
            self.data.to_netcdf(self.ncdf_file)
        try:
            self.grid = pv.read(self.vtm_file)
        except FileNotFoundError:
            self.build_grid()
            self.grid.save(self.vtm_file)

    def read_excel(self):
        """Read warm TFC1 centerline data from excel sheet."""
        self.data = xarray.Dataset(
            coords=dict(loadcase=['referance'],
                        coil=['TF1'], position=['x', 'y', 'z']))
        with pandas.ExcelFile(self.xlsx_file, engine='openpyxl') as xls:
            for i, sheet_name in enumerate(xls.sheet_names):
                name = f'dp{i}'
                index = f'index_{i}'
                double_pancake = self.read_sheet(xls, sheet_name)
                self.data[name] = xarray.DataArray(
                    1e-3*double_pancake.to_numpy(), dims=[index, 'position'],
                    coords={index: double_pancake.index})

    def read_sheet(self, xls, sheet_name):
        """Read excel worksheet."""
        sheet = pandas.read_excel(xls, sheet_name, usecols=[2, 3, 4])
        columns = {'X Coord': 'x', 'Y Coord': 'y', 'Z Coord': 'z'}
        sheet.rename(columns=columns, inplace=True)
        return sheet

    def build_grid(self):
        """Build pyvisa grid."""
        grid = {f'TF1_dp{i}': pv.Spline(self.data[f'dp{i}'].values)
                for i in range(7)}
        self.grid = pv.MultiBlock(grid)


if __name__ == '__main__':

    cl = TFCenterLine()

    cl.grid.plot()
