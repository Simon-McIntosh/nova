from dataclasses import dataclass, field
import os

import pyvista as pv

from nova.definitions import root_dir
from nova.structural.plotter import Plotter
from nova.structural.TFC18 import TFC18
from nova.structural.uniformwindingpack import UniformWindingPack


@dataclass
class TFC(Plotter):

    assembly: str = 'v4'
    referance: str = 'v0'
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        self.directory = os.path.join(root_dir, 'data/Ansys/TFC18')
        self.load()

    def load(self):
        assembly = TFC18('TFC18', self.assembly, 'WP',
                         scenario={'TFonly': 2}).mesh
        referance = TFC18('TFC18', self.referance, 'WP',
                          scenario={'TFonly': 2}).mesh

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

    tfc = TFC()
    tfc.warp('delta')
    #tfc.animate()
