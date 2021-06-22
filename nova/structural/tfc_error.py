from dataclasses import dataclass, field
import os

import pyvista as pv

from nova.definitions import root_dir
from nova.structural.plotter import Plotter
from nova.structural.TFC18 import TFC18


@dataclass
class TFC(Plotter):

    assembly: str = 'v4'
    referance: str = 'v0'
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        self.directory = os.path.join(root_dir, 'data/Ansys/TFC18')
        self.load()

    def load(self):
        assembly = TFC18(self.assembly, 'WP', scenario={'TFonly': 2}).mesh
        referance = TFC18(self.referance, 'WP', scenario={'TFonly': 2}).mesh

        self.mesh = referance.copy()
        self.mesh.clear_arrays()
        self.mesh.points += referance['TFonly']
        self.mesh['TFonly'] = assembly['TFonly'] - referance['TFonly']

    def animate(self, max_factor=200):
        """Animate displacement."""
        filename = os.path.join(self.directory, f'{self.assembly}_delta')
        super().animate(filename, 'TFonly', view='iso', max_factor=max_factor)

if __name__ == '__main__':

    tfc = TFC()
    tfc.animate()
