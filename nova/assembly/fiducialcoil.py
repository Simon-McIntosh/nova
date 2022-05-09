
"""Build fiducial coilset."""
from dataclasses import dataclass, field
import os
from typing import Union

import numpy as np
import pyvista as pv

from nova.definitions import root_dir
from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducialdata import FiducialData
from nova.structural.morph import Morph
from nova.structural.plotter import Plotter


@dataclass
class FiducialCoil(Plotter):
    """Manage fiducial coilset."""

    name: str = 'fiducial'
    version: Union[str, int] = -1
    decimate: float = 0.95
    fiducialdata: FiducialData = field(default_factory=FiducialData)
    mesh: pv.PolyData = field(init=False, default_factory=pv.PolyData)
    folder: str = 'TFCgapsG10'
    file: str = 'k0'

    def __post_init__(self):
        """Load fiducial mesh."""
        self._append_version()
        self.load_mesh()

    def _append_version(self):
        """Append version to filename."""
        if self.version is None:
            return
        if self.version == -1:
            self.version = (self.fiducialdata.data.clone.values == -1).sum()
        self.name += f'_{self.version:2d}'

    @property
    def filename(self):
        """Return full mesh filepath."""
        return os.path.join(root_dir, 'data/Assembly/toroidal_fiducial',
                            f'{self.name}.vtk')

    def load_mesh(self):
        """Load fiducial vtk mesh."""
        try:
            self.mesh = pv.read(self.filename)
        except FileNotFoundError:
            self.build_cage()
            self.add_frozen()
            self.mesh.save(self.filename)
        self.mesh.name = self.name

    def load_surface_mesh(self, part, decimate=0.9):
        """Return decimated surface mesh."""
        mesh = AnsysPost(self.folder, self.file, part).mesh
        if decimate == 0:
            return mesh
        return mesh.decimate_boundary(decimate)

    def load_case(self, decimate=0.9):
        """Return TFC1 case."""
        case = pv.PolyData()
        for part in ['case_il', 'case_ol']:
            case = case + AnsysPost('TFCgapsG10', 'k0', part.upper()).mesh
        case = case.clip((-np.sin(np.pi/18), np.cos(np.pi/18), 0),
                         origin=(0, 0, 0))
        case = case.clip((np.sin(np.pi/18), np.cos(np.pi/18), 0),
                         origin=(0, 0, 0), invert=False)
        if decimate == 0:
            return case
        return case.decimate_boundary(decimate)

    def build_cage(self):
        """Add morphed toroidal field coil winding packs."""
        case = self.load_case(decimate=0)
        windingpack = self.load_surface_mesh('E_WP_1', decimate=0)
        for i in range(18):
            #morph_wp = Morph(self.fiducialdata.mesh.extract_cells(i),
            #                 windingpack,
            #                 neighbors=1, kernel='linear').mesh
            morph_wp = windingpack.copy()
            Morph(self.fiducialdata.mesh.extract_cells(i)).predict(morph_wp)
            sub_wp = morph_wp.decimate_boundary(self.decimate)
            sub_wp = sub_wp.interpolate(morph_wp)
            #morph_case = Morph(sub_wp, case, smoothing=10).mesh
            morph_case = case.copy()
            Morph(sub_wp).predict(morph_case)
            sub_case = morph_case.decimate_boundary(self.decimate)
            sub_case = sub_case.interpolate(morph_case)
            self.mesh += sub_wp
            self.mesh += sub_case
            windingpack.rotate_z(20, inplace=True)
            case.rotate_z(20, inplace=True)

    def add_frozen(self):
        """
        Add frozen parts to provide far-field suport to RBF interpolant.

            - 'e_pf_coils'
            - 'e_cs_coils'
            - 'cs_tie_plates'
            - 'gs_bottom_plate_connection'

        """
        for part in ['gs_bottom_plate_connection']:
            self._add_frozen(part, decimate=self.decimate)

    def _add_frozen(self, part, decimate):
        """Add frozen mesh to fiducial dataset."""
        mesh = self.load_surface_mesh(part.upper(), decimate)
        self._freeze(mesh)
        self.mesh += mesh

    def _freeze(self, mesh):
        """Set mesh displacment to zero."""
        mesh['delta'] = np.zeros((mesh.n_points, 3))


if __name__ == '__main__':

    fiducialcoil = FiducialCoil('fiducial', 10)
    #fiducialcoil.mesh = fiducialcoil.mesh.slice((0, 0, 1))
    #fiducialcoil.warp(500, opacity=1, displace='delta')

    base = AnsysPost('TFCgapsG10', 'k0', 'all')
    Morph(fiducialcoil.mesh).predict(base)
