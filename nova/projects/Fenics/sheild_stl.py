from dataclasses import dataclass, field
import os

import pyvista as pv
import vedo

from nova.definitions import root_dir
from nova.electromagnetic.framespace import FrameSpace
from nova.utilities.time import clock


@dataclass
class ShieldSector:
    """Manage shield sector."""

    sector: int = 4
    file: str = 'IWS_FM_PLATE'
    path: str = None
    sectors: list[int] = field(init=False,
                               default_factory=lambda: [2, 3, 4, 6])
    mesh: pv.PolyData = field(init=False, repr=False)
    geom: pv.PolyData = field(init=False, repr=False)
    frame: FrameSpace = field(init=False)

    def __post_init__(self):
        """Load datasets."""
        self.check_sector()
        if self.path is None:
            self.path = os.path.join(root_dir, 'input/geometry/ITER/shield')
        self.load_mesh()
        self.load_frame()

    def check_sector(self):
        """Check sector number."""
        if self.sector not in self.sectors:
            raise IndexError(f'sector {self.sector} not in {self.sectors}')

    @property
    def filename(self):
        """Return sector filename."""
        return f'{self.file}_S{self.sector}'

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.path, f'{self.filename}.vtk')

    @property
    def stl_file(self):
        """Return full stl filename."""
        return os.path.join(self.path, f'{self.filename}.stl')

    @property
    def cdf_file(self):
        """Return netCDF filename."""
        return os.path.join(self.path, f'{self.file}.nc')

    def load_mesh(self):
        """Load mesh."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.mesh = self.read_stl()

    def read_stl(self):
        """Read stl file."""
        mesh = pv.read(self.stl_file)
        mesh.save(self.vtk_file)
        return mesh

    def load_frame(self):
        """Load shield dataframe."""
        try:
            self.frame = FrameSpace().load(self.cdf_file, f'S{self.sector}')
        except FileNotFoundError:
            self.frame = self.read_frame()

    def read_frame(self):
        """Return dataframe read from vtk file."""
        mesh = vedo.Mesh(self.vtk_file)
        parts = mesh.splitByConnectivity(10000)
        frame = FrameSpace(label=f'fiS{self.sector}', body='panel', delim='_')
        tick = clock(len(parts), header='loading decimated convex hulls')
        for i, part in enumerate(parts):
            frame += dict(vtk=part.scale(1e-3).c(i),
                          part=f'fiS{self.sector}', ferritic=True)
            tick.tock()
        frame.store(self.cdf_file, f'S{self.sector}')
        return frame

    def plot(self):
        """Plot vtk mesh."""
        vedo.show(self.frame.vtk)


if __name__ == '__main__':

    shield = ShieldSector(2)
    shield.plot()
    #shield.read_frame()
    # shield.plot()
