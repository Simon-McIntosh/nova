"""Manage ferromagnetic insert geometry."""
from dataclasses import dataclass, field
import os
from typing import ClassVar

import sklearn.cluster
import vedo

from nova.definitions import root_dir
from nova.electromagnetic.framespace import FrameSpace
from nova.utilities.time import clock


@dataclass
class ShieldBase:
    """Manage libary of shield sector instances."""

    sector: int
    avalible: ClassVar[list[int]] = [2, 3, 4, 6]

    def __post_init__(self):
        """Check sector avalibility."""
        self.check_avalible()

    def check_avalible(self):
        """Check avaliblity of requested sector."""
        if self.sector not in self.avalible:
            raise IndexError(f'sector {self.sector} not in {self.avalible}')


@dataclass
class ShieldDir:
    """Manage shield filepath."""

    file: str = 'IWS_FM_PLATE'
    path: str = None

    def __post_init__(self):
        """Define file paths."""
        if self.path is None:
            self.path = os.path.join(root_dir, 'input/geometry/ITER/shield')

    @property
    def cdf_file(self):
        """Return netCDF filename."""
        return os.path.join(self.path, f'{self.file}.nc')


@dataclass
class ShieldCad(ShieldDir, ShieldBase):
    """Manage shield sector stl translation."""

    mesh: vedo.Mesh = field(init=False, repr=False)
    frame: FrameSpace = field(init=False, repr=False)

    def __post_init__(self):
        """Load datasets."""
        super().__post_init__()
        self.mesh = self.load_mesh()
        self.frame = self.load_frame()

    @property
    def data(self):
        """Return mesh and frame data."""
        return self.mesh, self.frame

    @property
    def frame_metadata(self):
        """Return frame metadata."""
        return dict(label=f'fi{self.sector}', body='panel', delim='_')

    def frame_data(self, vtk: vedo.Mesh):
        """Return frame data."""
        return dict(vtk=vtk, part=f'fi{self.sector}', ferritic=True)

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.path, f'{self.file}_S{self.sector}.vtk')

    @property
    def stl_file(self):
        """Return full stl filename."""
        file = os.path.join(self.path, f'{self.file}_S{self.sector}.stl')
        if not os.path.isfile(file):
            raise FileNotFoundError(f'stl file {file} not found')
        return

    def load_mesh(self):
        """Return mesh."""
        if os.path.isfile(self.vtk_file):
            return vedo.Mesh(self.vtk_file)
        return self.build_mesh()

    def build_mesh(self):
        """Read stl file."""
        mesh = vedo.Mesh(self.stl_file).scale(1e-3)
        self.store_mesh(mesh)
        return mesh

    def store_mesh(self, mesh):
        """Write mesh to file."""
        mesh.write(self.vtk_file)

    def load_frame(self):
        """Return shield dataframe."""
        try:
            return FrameSpace().load(self.cdf_file, f'S{self.sector}')
        except (FileNotFoundError, OSError):
            return self.build_frame()

    def build_frame(self):
        """Return dataframe read from vtk file."""
        parts = self.mesh.splitByConnectivity(10000)
        frame = FrameSpace(**self.frame_metadata)
        tick = clock(len(parts), header='loading decimated convex hulls')
        for i, part in enumerate(parts):
            frame += self.frame_data(part.c(i))
            tick.tock()
        self.store_frame(frame)
        return frame

    def store_frame(self, frame):
        """Write frame to netCDF file."""
        mode = 'a' if os.path.isfile(self.cdf_file) else 'w'
        frame.store(self.cdf_file, f'S{self.sector}', mode=mode)

    def plot(self):
        """Plot vtk mesh."""
        vedo.show(self.frame.vtk)


@dataclass
class ShieldSector(ShieldCad):
    """Manage shield sector rotations."""

    avalible: ClassVar[list[int]] = range(1, 10)

    def __post_init__(self):
        """Load base data."""
        super().__post_init__()

    def build_mesh(self):
        """Build sector mesh via base rotation."""
        if self.sector in ShieldCad.avalible:
            return super().build_mesh()
        return self.rotate('mesh')

    def build_frame(self):
        """Build sector frame via base rotation."""
        if self.sector in ShieldCad.avalible:
            return super().build_frame()
        return self.rotate('frame')

    def rotate(self, attribute: str):
        """Perform base rotation."""
        if hasattr(self, attribute):
            return getattr(self, attribute)
        self._rotate()
        return getattr(self, attribute)

    def _rotate(self, base_sector=6):
        """Generate mesh and frame data via rotation from base."""
        degrees = 40*(self.sector-base_sector)
        base = ShieldCad(base_sector)
        self.mesh = self.rotate_mesh(base.mesh, degrees)
        self.frame = self.rotate_frame(base.frame, degrees)

    def rotate_mesh(self, base_mesh: vedo.Mesh, degrees: float):
        """Return rotated mesh and save output to vtk file."""
        mesh = base_mesh.clone().rotate(degrees, axis=(0, 0, 1),
                                        point=(0, 0, 0))
        self.store_mesh(mesh)
        return mesh

    def rotate_frame(self, base_frame: FrameSpace, degrees: float):
        """Rotate base frame and save output to netCDF file."""
        frame = FrameSpace(**self.frame_metadata)
        for vtk in base_frame.loc[:, 'vtk']:
            vtk = vtk.clone().rotate(degrees, axis=(0, 0, 1), point=(0, 0, 0))
            frame += self.frame_data(vtk)
        self.store_frame(frame)
        return frame


@dataclass
class ShieldSet(ShieldDir):
    """Manage complete ferritic insert dataset."""

    file: str = 'IWS_FM'
    mesh: vedo.Mesh = field(init=False, repr=False)
    frame: FrameSpace = field(init=False, repr=False)
    avalible: ClassVar[list[int]] = range(1, 10)

    def __post_init__(self):
        """Load shieldset."""
        super().__post_init__()
        self.frame = self.load_frame()

    def load(self):
        """Load shieldset."""

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.path, f'{self.file}.vtk')

    def load_mesh(self):
        """Return mesh."""
        if os.path.isfile(self.vtk_file):
            return vedo.Mesh(self.vtk_file)
        return self.build_mesh()

    def load_frame(self):
        """Return shield dataframe."""
        try:
            return FrameSpace().load(self.cdf_file)
        except (FileNotFoundError, OSError):
            return self.build_frame()

    def build_mesh(self):
        """Build complete ferritic sheildset vtk mesh."""
        tick = clock(len(self.avalible),
                     header='building ferritic insert mesh')
        mesh = []
        for i in self.avalible:
            sector = ShieldSector(i)
            mesh.append(sector.mesh)
            tick.tock()
        mesh = vedo.merge(mesh)
        mesh.write(self.vtk_file)
        return mesh

    def build_frame(self):
        """Build complete ferritic sheildset dataframe."""
        tick = clock(len(self.avalible),
                     header='building ferritic insert dataset')
        frame = []
        for i in self.avalible:
            sector = ShieldSector(i)
            frame.append(sector.frame)
            tick.tock()
        frame = FrameSpace().concatenate(*frame)
        frame.store(self.cdf_file, mode='w')

    @staticmethod
    def _cluster(frame: FrameSpace, eps_factor=1.5):
        """Cluster shield panels."""
        points = frame.loc[:, ['x', 'y', 'z']]
        eps = eps_factor*frame.dt.max()
        cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=1).fit(points)

        labels = set(cluster.labels_)




    def plot_mesh(self):
        """Plot vtk mesh."""
        vedo.show(self.mesh, *self.frame.vtk, axes=3)

    def plot_points(self):
        """Plot centroid point cloud."""
        vedo.show(vedo.Mesh(self.frame.loc[:, ['x', 'y', 'z']].values))


if __name__ == '__main__':

    shield = ShieldSet()

    shield._cluster(shield.frame[shield.frame])



    #shield.plot_points()
    #shield.frame.polyplot()
    #shield.plot_mesh()
