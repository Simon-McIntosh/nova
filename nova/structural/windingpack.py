"""Manage as-designed coil winding pack descriptors."""
from dataclasses import dataclass, field
import os

import pandas
import pyvista as pv
import xarray

from nova.definitions import root_dir


@dataclass
class WindingPack:
    """Load referance winding pack centerlines from file."""

    file: str
    directory: str = None
    data: xarray.Dataset = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

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
    def vtk_file(self):
        """Return vtk filepath."""
        return self.filepath('vtk')

    def load_single(self):
        """Load centerline data."""
        try:
            self.data = xarray.open_dataset(self.ncdf_file)
        except FileNotFoundError:
            self.read_excel()
            self.data.to_netcdf(self.ncdf_file)
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.pattern_TF()
            self.mesh.save(self.vtk_file)

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

    def pattern_TF(self):
        """Build pyvisa mesh."""
        mesh = {f'TF1_dp{dp_index}':
                pv.Spline(self.data[f'dp{dp_index}'].values)
                for dp_index in range(7)}
        for TF_index in range(2, 19):  # TF coils 2-18
            for dp_index in range(7):
                referance = f'TF1_dp{dp_index}'
                target = f'TF{TF_index}_dp{dp_index}'
                mesh[target] = mesh[referance].copy()
                mesh[target].rotate_z(360 * (TF_index-1) / 18)
        self.mesh = pv.MultiBlock(mesh).combine()

    def plot(self):
        """Plot winding pack."""
        plotter = pv.Plotter()
        plotter.add_mesh(wp.mesh)
        plotter.show()


if __name__ == '__main__':

    wp = WindingPack('TFC1_CL')
    wp.mesh.plot()
