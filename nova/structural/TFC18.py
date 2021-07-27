"""Apply structural deformation to TF coilset."""
from dataclasses import dataclass, field
import os

import numpy as np
import pandas
import pyvista as pv
import xarray

from nova.structural.ansyspost import AnsysPost
from nova.structural.clusterturns import ClusterTurns
from nova.structural.datadir import AnsysDataDir
from nova.structural.plotter import Plotter
from nova.structural.windingpack import WindingPack
from nova.structural.uniformwindingpack import UniformWindingPack


@dataclass
class TFC18(AnsysDataDir, Plotter):
    """Post-process Ansys output from F4E's 18TF coil model."""

    cluster: int = 1
    scenario: dict[str, int] = field(
        default_factory=lambda: dict(cooldown=1, TFonly=2))
    ansys: pv.PolyData = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load database."""
        super().__post_init__()
        self.subset = 'WP'
        self.load()

    def __str__(self):
        """Return Ansys model descriptor."""
        return AnsysPost(*self.metadata).__str__()

    @property
    def vtk_file(self):
        """Return vtk file path."""
        return os.path.join(self.directory, f'{self.file}.vtk')
    
    @property 
    def csv_file(self):
        """Return csv filename."""
        return os.path.join(self.directory, 
                            f'{self.file}_{self.cluster}loop.csv')

    def load(self):
        """Load vtm data file."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.load_ansys()
            self.load_mesh()
        if self.cluster:
            self.mesh = ClusterTurns(self.mesh, self.cluster).mesh

    def load_ansys(self):
        """Load ansys vtk mesh."""
        ansys = AnsysPost(self.folder, self.file, self.subset).mesh
        self.ansys = ansys.copy()
        self.ansys.clear_point_arrays()
        for scn in self.scenario:
            self.ansys[scn] = ansys[f'disp-{self.scenario[scn]}']

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
            self.mesh[scn] = mesh[scn]
        self.mesh['turns'] = mesh['turns']
        self.mesh.save(self.vtk_file)

    def interpolate_coils(self, source, target, sharpness=3, radius=1.5,
                          n_cells=7):
        """Retun interpolated mesh."""
        return source.interpolate(target, sharpness=sharpness, radius=radius,
                                  strategy='closest_point')
    
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

    def plot(self):
        """Plot warped shape."""
        self.warp('TFonly-cooldown')

    def animate(self):
        """Animate displacement."""
        filename = os.path.join(self.directory, self.file)
        super().animate(filename, 'TFonly-cooldown', view='xy')


if __name__ == '__main__':

    tf = TFC18('TFC18', 'v4', cluster=5)
        
    #tf.to_dataframe()

    #tf.mesh['TFonly-cooldown'] = tf.mesh['TFonly'] - tf.mesh['cooldown']
    #tf.warp('TFonly-cooldown', factor=120)
    
    #tf.animate()
