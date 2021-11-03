"""Implement interface between xarray / ansys and csv dataframes."""
from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import pyvista as pv
from scipy.spatial.transform import Rotation
import xarray

from nova.structural.datadir import DataDir
from nova.structural.fiducialdata import FiducialData
from nova.structural.fiducialerror import FiducialError
from nova.structural.plotter import Plotter


@dataclass
class CoilCage(DataDir, Plotter):
    """Manage deformed CCL data."""

    folder: str = 'ccl'
    file: str = 'fit'
    data_dir: str = 'data/Assembly/TFC18'
    mesh: pv.PolyData = field(init=False, repr=False)

    def from_xarray(self, data: xarray.Dataset):
        """Load mesh data from xarray."""
        data = data.sortby('location')
        self.mesh = pv.PolyData()
        centerline = pv.Spline(data.centerline)
        for location in data.location[:18].values:
            coil = centerline.copy()
            coil.rotate_z(20*location)
            midplane = coil.slice(normal='z', origin=(0, 0, 0))
            coil['ID'] = [midplane.points[0]]
            coil['OD'] = [midplane.points[1]]
            self.mesh = self.mesh.merge(coil, merge_points=False)
        for cell_data in ['coil', 'origin', 'clone', 'location']:
            self.mesh[cell_data] = data[cell_data][:18]
        label = [f'{coil:02d}' if clone == -1 else f'{coil:02d}<{clone:02d}'
                 for coil, clone in zip(self.mesh['coil'], self.mesh['clone'])]
        self.mesh['label'] = label
        self.mesh['cluster'] = np.full(self.mesh.n_cells, 1)
        self.mesh['nturn'] = np.full(self.mesh.n_cells, 134)
        return self

    def add_scenario(self, data: xarray.Dataset, scenario: str,
                     source='fit_delta'):
        """Add fit_delta as scenario to vtk mesh."""
        delta = data[source][:18].values
        for i in range(18):
            delta[i] = Rotation.from_euler('z', i*np.pi/9).apply(delta[i])
        self.mesh[scenario] = delta.reshape(-1, 3)

    def label_coils(self, plotter, location='OD'):
        """Add coil labels."""
        plotter.add_point_labels(self.mesh[location][:18],
                                 self.mesh['label'][:18], font_size=20)

    def to_dataframe(self, scenario):
        """Save mesh[scenario] as dataframe."""
        mesh = self.mesh.copy()
        mesh.points += mesh[scenario]
        frames = list()
        for cell in range(mesh.n_cells):
            points = mesh.cell_points(cell)
            n_seg = len(points) - 1
            coil = int(mesh['coil'][cell])
            cluster = int(mesh['cluster'][cell])
            nturn = int(mesh['nturn'][cell])
            cpoint = (points[1:]+points[:-1]) / 2  # centerpoint
            vector = points[1:] - points[:-1]
            data = dict(coil=np.full(n_seg, f'TF{coil}'),
                        cluster=np.full(n_seg, cluster),
                        nturn=np.full(n_seg, nturn),
                        x=cpoint[:, 0], y=cpoint[:, 1], z=cpoint[:, 2],
                        dx=vector[:, 0], dy=vector[:, 1], dz=vector[:, 2])
            frames.append(pandas.DataFrame(data))
        frame = pandas.concat(frames)
        frame.to_csv(self.csv_file(scenario), index=False)

    def load_dataset(self):
        """Load vtk data file."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.build_dataset()

    def build_dataset(self):
        """Build fiducial dataset."""
        data = FiducialData(fill=True, sead=2025).data
        error = FiducialError(data)
        self.from_xarray(error.data)
        self.add_scenario(error.data, 'asbuilt', source='centerline_delta')
        for label, weight in zip(['inboard', 'equal', 'outboard'],
                                 [-1, 0, 1]):
            error.fit_coilset(weight)
            self.add_scenario(error.data, label)
        self.mesh.save(self.vtk_file)

    def csv_file(self, scenario: str):
        """Return csv filename."""
        return os.path.join(self.directory, f'{self.file}_{scenario}.csv')

    @property
    def vtk_file(self):
        """Return vtk filename."""
        return os.path.join(self.directory, f'{self.file}.vtk')

    def export(self):
        """Export datasets to csv files."""
        for scenario in ['asbuilt', 'inboard', 'equal', 'outboard']:
            self.to_dataframe(scenario)

    def plot(self):
        """Plot warped coilcage with labels."""
        plotter = pv.Plotter()
        self.warp(500, plotter=plotter)
        self.label_coils(plotter)
        plotter.show_axes()
        plotter.show()


if __name__ == '__main__':

    cage = CoilCage()
    cage.load_dataset()
    cage.export()

    #cage.plot()
