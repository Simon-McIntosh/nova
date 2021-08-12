"""Apply structural deformation to TF coilset."""
from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import pathlib
import pyvista as pv
import xarray

from nova.structural.ansyspost import AnsysPost
from nova.structural.clusterturns import ClusterTurns
from nova.structural.datadir import DataDir
from nova.structural.plotter import Plotter
from nova.structural.windingpack import WindingPack
from nova.structural.uniformwindingpack import UniformWindingPack
from nova.utilities.time import clock


@dataclass
class TFC18(DataDir, Plotter):
    """Post-process Ansys output from F4E's 18TF coil model."""

    cluster: int = 1
    scenario: dict[str, int] = field(
        default_factory=lambda: dict(cooldown=1, TFonly=2, SOD=3, EOB=4))
    ansys: pv.PolyData = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load database."""
        super().__post_init__()
        self.subset = 'WP'
        self.load()

    def __str__(self):
        """Return Ansys model descriptor (takes time - remote mount)."""
        return AnsysPost(*self.ansys_args).__str__()

    def reload(self, file):
        """Reload source file."""
        self.file = file
        self.__post_init__()

    def load_ensemble(self):
        """Load ensemble dataset and store reduced data in vtk format."""
        _file = self.file
        paths = list(pathlib.Path(self.rst_folder).rglob('*.rst'))
        files = [file for path in paths if not
                 os.path.isfile(self.vtk_file.replace(
                     self.file, file := path.name[:-4]))]
        nfiles = len(files)
        tick = clock(nfiles, header=f'loading {nfiles} *.rst files [{files}]')
        for file in files:
            self.reload(file)
            tick.tock()
        self.reload(_file)

    def load(self):
        """Load vtm data file."""
        try:
            self.mesh = pv.read(self.ccl_file)
        except FileNotFoundError:
            self.load_ansys()
            self.load_mesh()
        if self.cluster:
            self.mesh = ClusterTurns(self.mesh, self.cluster).mesh

    def load_ansys(self):
        """Load ansys vtk mesh."""
        ansys = AnsysPost(*self.args).mesh
        self.ansys = ansys.copy()
        self.ansys.clear_point_arrays()
        for scn in self.scenario:
            try:
                self.ansys[scn] = ansys[f'disp-{self.scenario[scn]}']
            except KeyError:
                pass

    def load_windingpack(self):
        """Load conductor windingpack."""
        if self.cluster is not None:
            return UniformWindingPack().mesh
        return WindingPack('TFC1_CL').mesh

    def load_mesh(self):
        """Load referance windingpack ccl."""
        self.mesh = self.load_windingpack()
        self.mesh = self.interpolate_coils(self.mesh, self.ansys)
        mesh = self.mesh.copy()
        self.mesh.clear_arrays()
        for scn in self.scenario:
            try:
                self.mesh[scn] = mesh[scn]
            except KeyError:
                pass
        try:
            self.mesh['turns'] = mesh['turns']
        except KeyError:
            pass
        self.mesh.save(self.ccl_file)

    def interpolate_coils(self, source, target, sharpness=3, radius=1.5,
                          n_cells=7):
        """Retun interpolated mesh."""
        return source.interpolate(target, sharpness=sharpness, radius=radius,
                                  strategy='closest_point')

    @property
    def csv_file(self):
        """Return csv filename."""
        return os.path.join(self.directory,
                            f'{self.file}_{self.cluster}loop.csv')

    def to_dataframe(self):
        """Return mesh as dataframe."""
        mesh = self.mesh.copy()
        mesh.points += mesh['TFonly']
        frames = list()
        for cell in range(mesh.n_cells):
            points = mesh.cell_points(cell)
            n_seg = len(points) - 1
            coil = int(mesh['coil'][cell])
            cluster = int(mesh['cluster'][cell])
            nturn = int(mesh['nturn'][cell])
            cpoint = (points[1:]+points[:-1]) / 2  # centerpoint
            vector = points[1:] - points[:-1]
            data = dict(coil=np.full(n_seg, f'TF{coil+1}'),
                        cluster=np.full(n_seg, cluster),
                        nturn=np.full(n_seg, nturn),
                        x=cpoint[:, 0], y=cpoint[:, 1], z=cpoint[:, 2],
                        dx=vector[:, 0], dy=vector[:, 1], dz=vector[:, 2])
            frames.append(pandas.DataFrame(data))
        frame = pandas.concat(frames)
        print(frame)

        frame.to_csv(self.csv_file, index=False)

        #index = range(len(self.mesh.cell_points(0))-1)

        #dataset = xarray.Dataset(
        #    coords=dict(scenario=['TF-only'],

        #                                     )
        #def to_xarray(self):
        #print(xarray.Dataset({'v4': frame}))

    def diff(self, displace: str, reference: str='TFonly'):
        """Diffrence array and return name."""
        name = f'{displace}-{reference}'
        if name not in self.mesh.array_names:
            self.mesh[name] = self.mesh[displace] - self.mesh[reference]
        return name

    def plot(self, displace: str, referance='TFonly', factor=80):
        """Plot warped shape."""
        self.warp(self.diff(displace, referance), factor=factor)

    def animate(self, displace: str, view='xy'):
        """Animate displacement."""
        filename = os.path.join(self.directory, self.file)
        super().animate(filename, self.diff(displace), view=view,
                        max_factor=80)


if __name__ == '__main__':

    tf = TFC18('TFCgapsG10', 'k0', cluster=None)

    tf.load_ensemble()
    #tf.mesh['TFonly-cooldown'] = tf.mesh['TFonly'] - tf.mesh['cooldown']

    #tf.to_dataframe()
    #tf.plot('TFonly', 'cooldown', factor=50)
    #
    #tf.warp('TFonly-cooldown', factor=120)

    #tf.animate('EOB', view='iso')
