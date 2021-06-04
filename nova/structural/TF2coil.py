
from dataclasses import dataclass
import os

import numpy as np

from ansys.mapdl.core import launch_mapdl
from ansys.mapdl import core as pymapdl
from contextlib import contextmanager

from nova.definitions import root_dir


@dataclass
class ToroidalFieldCoil:
    """Build TF two coil model."""

    macro: str = 'TF2coil'

    def __post_init__(self):
        """Resolve data path."""
        self.directory = os.path.join(root_dir, 'data/Ansys')
        self.macro = os.path.join(root_dir, f'input/APDL/{self.macro}')

    def convert_macros(self):

        pymapdl.convert_script(f'{self.macro}/tf_mkdir.mac',
                               f'{self.macro}/tf_mkdir.py',
                               macros_as_functions=True,
                               auto_exit=False)


    def run(self):
        mapdl = launch_mapdl(mode='corba', override=True,
                             start_instance=False)

        length = 10

        # simple 3D beam
        mapdl.clear()
        mapdl.prep7()
        mapdl.mp("EX", 1, 70000)
        mapdl.mp("NUXY", 1, 0.3)
        mapdl.csys(0)
        mapdl.blc4(0, 0, 0.5, 2, length)
        mapdl.et(1, "SOLID186")
        mapdl.type(1)
        mapdl.keyopt(1, 2, 1)
        mapdl.desize("", 100)

        mapdl.vmesh("ALL")
        #mapdl.eplot()

        # fixed constraint
        mapdl.nsel("s", "loc", "z", 0)
        mapdl.d("all", "ux", 0)
        mapdl.d("all", "uy", 0)
        mapdl.d("all", "uz", 0)

        # arbitrary non-uniform load
        mapdl.nsel("s", "loc", "z", length)
        mapdl.f("all", "fz", 1)
        mapdl.f("all", "fy", 10)
        mapdl.nsel("r", "loc", "y", 0)
        mapdl.f("all", "fx", 10)
        mapdl.allsel()
        mapdl.run("/solu")
        mapdl.solve()

        mapdl.cdread('db', 'TF_COILS_V9_4')
        #mapdl.post_processing.plot_nodal_displacement(lighting=False, show_edges=True)
        mapdl.exit()
        #self.mapdl = mapdl



if __name__ == '__main__':

    tf = ToroidalFieldCoil()
    tf.run()
