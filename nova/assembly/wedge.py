"""Visulize non-parallel gap simulations."""
from dataclasses import dataclass, field
import os
from typing import Union

import numpy as np
import pyvista as pv

from nova.assembly.ansysvtk import AnsysVTK
from nova.assembly.gap import WedgeGap
from nova.assembly.plotter import Plotter
from nova.definitions import root_dir


@dataclass
class Wedge(Plotter):
    """Post-process Ansys output from non-parallel gap models."""

    file: str
    subset: Union[str, list[str]] = field(
        default_factory=lambda: ['case_il', 'case_ol'])
    factor: float = 120.
    folder: str = 'TFCgapsG10'
    datapath: str = 'data/Assembly'
    mesh: pv.UnstructuredGrid = field(init=False, repr=False)

    def __post_init__(self):
        """Load simulation vtk file."""
        self.path = os.path.join(root_dir, self.datapath)
        if not isinstance(self.subset, list):
            self.subset = [self.subset]
        try:
            self.load()
        except FileNotFoundError:
            self.build()

    @property
    def filename(self):
        """Return mesh filename."""
        name = f'wedge_{self.file}_' + '_'.join(self.subset)
        name += f'_{self.factor}.vtk'
        return os.path.join(self.path, name)

    def load(self):
        """Load rotated mesh from file. Build if not found."""
        self.mesh = pv.read(self.filename)

    def build(self):
        """Build rotated mesh."""
        vtk = pv.UnstructuredGrid()
        for subset in self.subset:
            vtk += AnsysVTK(file=self.file, subset=subset).mesh
        gap_data = WedgeGap().data.sel(simulation=self.file)
        self.mesh = pv.UnstructuredGrid()
        for i in range(18):
            mesh = vtk.split_bodies()[i]
            center = mesh.center_of_mass()
            phi = np.arctan2(center[1], center[0])
            if phi < -np.pi/18:
                phi += 2*np.pi
            index = int(np.round(9*phi / np.pi))
            phi = index*np.pi / 9
            vector = np.array([np.cos(phi), np.sin(phi), 0])
            point = 1e-3*gap_data.radius * vector
            phi = gap_data.rotate.data[index, 0]
            roll = gap_data.rotate.data[index, 1]
            yaw = gap_data.rotate.data[index, 2]
            mesh.rotate_vector(
                vector, self.factor*180/np.pi*roll, point,
                transform_all_input_vectors=False, inplace=True)
            mesh.rotate_z(self.factor*180/np.pi*yaw, point,
                          transform_all_input_vectors=False, inplace=True)
            mesh.rotate_z(self.factor*180/np.pi*phi,
                          transform_all_input_vectors=False, inplace=True)
            self.mesh += mesh
        self.mesh['TFonly-cooldown'] = \
            self.mesh['TFonly'] - self.mesh['cooldown']
        self.mesh.save(self.filename)

    def slice_z(self):
        """Slice mesh."""
        self.mesh = wedge.mesh.clip_box([-5, 5, -5, 5, -0.1, 0], invert=False)

    def clip_box(self):
        """Clip mesh."""
        self.mesh = wedge.mesh.clip_box([-15, 0, -15, 0, -15, 15],
                                        invert=False)

    def warp(self):
        """Plot warped mesh."""
        super().warp(self.factor)

    def animate(self, name: str, view: str, zoom=1.3, opacity=0.75):
        """Make animation."""
        filename = os.path.join(self.path, f'{self.file}_{name}')
        super().animate(filename, 'TFonly-cooldown', view=view,
                        max_factor=self.factor, zoom=zoom, opacity=opacity)


if __name__ == '__main__':

    wedge = Wedge('w4', factor=50)

    #wedge.slice_z()
    #wedge.clip_box()

    wedge.mesh = wedge.mesh.clip_box([-15, 15, -15, 15, -15, 0], invert=False)
    #wedge.mesh = wedge.mesh.clip_box([-5, 5, -5, 5, -0.5, 0], invert=False)

    wedge.animate('corrected_full', view='xy', zoom=1.3, opacity=0.75)

    #wedge.warp()

    #wedge.animate('wp_quarter', 'iso')
    #reference = CCL('TFCgapsG10', 'v0')
    #wedge.mesh = wedge.mesh.clip('z', invert=True)
    #wedge

    #wedge.mesh - pv.UnstructuredGrid(reference.mesh)

    #wedge.mesh = wedge.mesh.slice('z')

    #wedge
