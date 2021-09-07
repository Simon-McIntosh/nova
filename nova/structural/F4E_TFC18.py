
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
class F4E_Data:
    """Load F4E data from 2016 study.

    ITER_D_TRTG3L v1.0
    TF Magnet Assembly Analysis:
        FE Analysis simulating non constant gap between inner legs
    """

    filename: str = field(init=False, default='CCL_Deformed_Coils-nm')
    scenarios: list[str] = field(init=False, default_factory=lambda:
                                 ['V0', 'V1', 'V3', 'V4'])
    datasheet: pandas.DataFrame = field(init=False,
                                        default_factory=pandas.DataFrame)

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
            self.load_geometry()
            self.load_dataset()
            self.mesh.save(self.vtk_file)

    @property
    def scenario(self):
        """Return scenario identifier."""
        return self.datasheet.index.name

    @scenario.setter
    def scenario(self, scenario):
        """Load scenario from excel file."""
        self.datasheet = pandas.read_excel(self.excel_file, scenario)
        self.datasheet.index.name = scenario

    def load_geometry(self):
        """Load single coil referance geometry."""
        self.scenario = self.scenarios[0]
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
        for scenario in self.scenarios:
            self.scenario = scenario
            for coil in range(1, 19):
                self.mesh[f'{scenario}-TFC{coil}'] = self.load_data(coil)

    def interpolate(self, mesh: pv.PolyData):
        """Interpolate coil displacment onto new vtk mesh."""
        for array in self.mesh.array_names:
            if array == 'arc_length':
                continue
            mesh[array] = scipy.interpolate.interp1d(
                self.mesh['arc_length'], self.mesh[array],
                axis=0)(mesh['arc_length'])
        return mesh

    def pattern_mesh(self):
        """Pattern TF coils."""
        TFC1 = self.mesh.copy()
        TFC1.clear_point_arrays()
        mesh = pv.PolyData()
        for i in range(18):
            coil = TFC1.copy()
            coil['V4'] = self.mesh[f'V4-TFC{i+1}']
            coil.rotate_z(360*i / 18)
            mesh += coil
        return mesh


if __name__ == '__main__':

    f4e = F4E_Data()

    from nova.structural.centerline import CenterLine

    cl = CenterLine()

    mesh = f4e.interpolate(cl.mesh)
    #mesh.plot()
    mesh = f4e.pattern_mesh().tube(0.1)
    #mesh.plot()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=None, color='w',
                     smooth_shading=True)
    warp = mesh.warp_by_vector('V4', factor=120)
    plotter.add_mesh(warp, scalars='V4')
    plotter.show()

'''
@dataclass
class F4E_CCL(UniformWindingPack, Plotter):

    cluster: int = 5

    def __post_init__(self):
        """Initalise TF coil-cage mesh."""
        super().__post_init__()
        self.cluster_turns()
        self.mesh_tube()

    def cluster_turns(self):
        """Apply k-means cluster algoritum to TFC turns."""
        if self.cluster is None:
            return
        self.mesh = ClusterTurns(self.mesh, self.cluster).mesh

    def mesh_tube(self, radius):
        """Generate tubes from conductor lines."""
        if self.cluster > 10:
            return
        self.mesh = self.mesh.tube(0.5/self.cluster)


        #self.mesh['V0'] = np.append(delta, delta.iloc[:1, :], axis=0)
        #self.mesh = self.mesh.tube(radius=0.5, n_sides=9)
        #self.mesh.tube(radius=0.1).plot(smooth_shading=True)


if __name__ == '__main__':

    ccl = F4E_CCL()
    #ccl.warp('V0')
'''
