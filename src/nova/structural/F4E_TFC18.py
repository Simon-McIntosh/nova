"""Manage 2016 F4E TFC data."""
from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import pyvista as pv
import scipy.interpolate

from nova.definitions import root_dir
from nova.structural.clusterturns import ClusterTurns
from nova.structural.plotter import Plotter
from nova.structural.uniformwindingpack import UniformWindingPack


@dataclass
class F4E_Data(Plotter):
    """Load F4E data from 2016 study.

    ITER_D_TRTG3L v1.0
    TF Magnet Assembly Analysis:
        FE Analysis simulating non constant gap between inner legs
    """

    filename: str = field(init=False, default='CCL_Deformed_Coils-nm')
    gaps: list[str] = field(init=False, default_factory=lambda:
                            ['V0', 'V1', 'V3', 'V4'])
    datasheet: pandas.DataFrame = field(init=False,
                                        default_factory=pandas.DataFrame)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Init filename and load referance geometory."""
        self.load()

    @property
    def excel_file(self):
        """Return excel data filepath."""
        return os.path.join(root_dir, 'input/geometry/ITER',
                            self.filename + '.xlsx')

    @property
    def vtk_file(self):
        """Return vtk data filepath."""
        return os.path.join(root_dir, 'data/Assembly',
                            self.filename + '.vtk')

    def load(self):
        """Load mesh from vtk datafile."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.reload()

    def reload(self):
        """Reload datafile."""
        self.load_geometry()
        self.load_dataset()
        self.load_cage()
        self.diff('v3', 'v0')
        self.mesh.save(self.vtk_file)

    @property
    def gap(self):
        """Return gap identifier."""
        return self.datasheet.index.name

    @gap.setter
    def gap(self, gap):
        """Load gap listing from excel file."""
        self.datasheet = pandas.read_excel(self.excel_file, gap)
        self.datasheet.index.name = gap

    def load_geometry(self):
        """Load single coil referance geometry."""
        self.gap = self.gaps[0]
        points = self.datasheet.loc[:, ['X', 'Y', 'Z']]
        points = self.close_loop(points)
        self.mesh = pv.Spline(points)
        self.mesh['arc_length'] /= self.mesh['arc_length'][-1]

    @staticmethod
    def close_loop(data: pandas.DataFrame, axis=0):
        """Return data set with startpoint appended to end."""
        return pandas.concat([data, data.iloc[:1, :]], axis=axis,
                             ignore_index=True)

    def load_data(self, coil: int):
        """Return displacment for single coil and store in pyvista mesh."""
        data = self.datasheet.loc[:, [f'R-TF{coil}', f'T-TF{coil}',
                                      f'Z-TF{coil}']]
        return self.close_loop(data)

    def load_dataset(self):
        """Load full ensemble dataset."""
        for gap in self.gaps:
            self.gap = gap
            for coil in range(1, 19):
                self.mesh[f'{gap}-TFC{coil}'] = self.load_data(coil)

    def interpolate(self, mesh: pv.PolyData):
        """Interpolate coil displacments onto new vtk mesh."""
        for array in self.mesh.array_names:
            if array == 'arc_length':
                continue
            mesh[array] = scipy.interpolate.interp1d(
                self.mesh['arc_length'], self.mesh[array],
                axis=0)(mesh['arc_length'])
        return mesh

    def pattern(self, mesh: pv.PolyData):
        """Pattern TF coils."""
        TFC1 = mesh.copy()
        mesh = pv.PolyData()
        for i in range(18):
            coil = TFC1.copy()
            coil.clear_data()
            coil['arc_length'] = TFC1['arc_length']
            for gap in self.gaps:
                coil[gap.lower()] = TFC1[f'{gap}-TFC{i+1}']
            coil.rotate_z(360*i / 18, transform_all_input_vectors=True)
            mesh += coil
        return mesh

    @staticmethod
    def load_reference():
        """Return reference TFC1 single turn centerline."""
        mesh = ClusterTurns(UniformWindingPack().mesh, 1).mesh.extract_cells(0)
        mesh = pv.Spline(np.append(mesh.points, mesh.points[:1], axis=0))
        mesh['arc_length'] /= mesh['arc_length'][-1]
        return mesh

    def load_cage(self):
        """Interpolate excel data to referance ccl and pattern coils."""
        mesh = self.load_reference()
        mesh = self.interpolate(mesh)
        self.mesh = self.pattern(mesh)


if __name__ == '__main__':

    f4e = F4E_Data()
    #f4e.reload()
    #f4e.mesh.save(f4e.vtk_file)
    #f4e.warp()
