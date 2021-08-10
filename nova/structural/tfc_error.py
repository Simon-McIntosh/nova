from dataclasses import dataclass, field
import os

import pyvista as pv

from nova.definitions import root_dir
from nova.structural.plotter import Plotter
from nova.structural.TFC18 import TFC18
from nova.structural.uniformwindingpack import UniformWindingPack


@dataclass
class TFC(Plotter):
    """Construct error displacement fields for TF coilset."""

    folder: str
    data_dir: str = '//io-ws-ccstore1/ANSYS_Data/mcintos'
    loadcase: tuple[str] = ('k0', 'c2')
    scenario: dict[str, int] = field(default_factory=lambda: {'TFonly': 2})
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load analysis data."""
        #self.directory = os.path.join(root_dir, 'data/Ansys/TFC18')
        self.directory = self.data_dir
        self.load_ansys_data()

    def load_ansys_data(self):
        referance = TFC18(self.folder, self.loadcase[0],
                          scenario=self.scenario, data_dir=self.data_dir).mesh
        assembly = TFC18(self.folder, self.loadcase[1],
                         scenario=self.scenario, data_dir=self.data_dir).mesh

        self.mesh = referance.copy()
        self.mesh.clear_arrays()
        self.mesh['referance'] = referance['TFonly']
        self.mesh['assembly'] = assembly['TFonly']
        self.mesh['delta'] = self.mesh['assembly'] - self.mesh['referance']

    def animate(self, max_factor=200):
        """Animate displacement."""
        filename = os.path.join(self.directory, f'{self.assembly}_delta')
        super().animate(filename, 'TFonly', view='iso', max_factor=max_factor)


    def cluster(self, n_cluster=1):
        clusters = UniformWindingPack().cluster(n_cluster)


if __name__ == '__main__':

    tfc = TFC('TFCgapsG10', data_dir='//io-ws-ccstore1/ANSYS_Data/mcintos',
              loadcase=('k0', 'c2'))

    #tfc = TFC('TFC18/parallel',
    #          data_dir='//io-ws-ccstore1/ANSYS_Data/mcintos',
    #          loadcase=('v0', 'v4'))

    tfc.warp('delta', factor=500)
    #tfc.animate()
