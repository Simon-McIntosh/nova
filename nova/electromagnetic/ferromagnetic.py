"""Manage ferromagnetic insert geometry."""
from dataclasses import dataclass, field
import os

import pandas
import pyvista as pv
import vedo

from nova.definitions import root_dir
from nova.electromagnetic.framespace import FrameSpace
from nova.utilities.time import clock


@dataclass
class ShieldSector:
    """Manage shield sector."""

    sector: int = 2
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

    def load_sector(self, sector: int):
        """Load sector data, return self."""
        self.sector = sector
        self.load_mesh()
        self.load_frame()
        return self

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
        if os.path.isfile(self.vtk_file):
            self.mesh = vedo.Mesh(self.vtk_file)
        else:
            self.mesh = self.read_stl()

    def read_stl(self):
        """Read stl file."""
        mesh = vedo.Mesh(self.stl_file).scale(1e-3)
        mesh.write(self.vtk_file)
        return mesh

    def load_frame(self):
        """Return shield dataframe."""
        try:
            self.frame = FrameSpace().load(self.cdf_file, f'S{self.sector}')
        except (FileNotFoundError, OSError):
            self.frame = self.read_frame()

    def read_frame(self):
        """Return dataframe read from vtk file."""
        parts = self.mesh.splitByConnectivity(10000)
        frame = FrameSpace(label=f'fi{self.sector}', body='panel', delim='_')
        tick = clock(len(parts), header='loading decimated convex hulls')
        for i, part in enumerate(parts):
            frame += dict(vtk=part.c(i),
                          part=f'fi{self.sector}', ferritic=True)
            tick.tock()
        mode = 'a' if os.path.isfile(self.cdf_file) else 'w'
        frame.store(self.cdf_file, f'S{self.sector}', mode=mode)
        return frame

    def plot(self):
        """Plot vtk mesh."""
        vedo.show(self.frame.vtk)


@dataclass
class FerriticInsert:
    """Manage complete ferritic insert dataset."""

    frame: FrameSpace = field(init=False, default_factory=FrameSpace)
    shield: ShieldSector = field(init=False, default_factory=ShieldSector)

    def __post_init__(self):
        """Load sectors."""
        for sector in self.shield.sectors:
            self.frame = pandas.concat(
                [self.frame, self.shield.load_sector(sector).frame])

    def plot(self):
        """Plot vtk mesh."""
        vedo.show(self.frame.vtk)


if __name__ == '__main__':

    fi = FerriticInsert()
    fi.plot()
    #sheild = ShieldSector(6)

    #print(fi.frame)
    #shield.read_frame()
    # shield.plot()
