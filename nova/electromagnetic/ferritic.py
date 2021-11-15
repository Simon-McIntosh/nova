"""Manage ferromagnetic insert geometry."""
from dataclasses import dataclass, field
import os
from string import digits
from typing import ClassVar, Union

import numpy as np
import sklearn.cluster
import vedo

from nova.definitions import root_dir
from nova.electromagnetic.framesetloc import FrameSetLoc
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

    def plot_mesh(self):
        """Plot vtk mesh."""
        vedo.show(self.mesh, *self.frame.vtk, axes=3)

    def plot_points(self):
        """Plot centroid point cloud."""
        vedo.show(vedo.Mesh(self.frame.loc[:, ['x', 'y', 'z']].values))


@dataclass
class Cluster:
    """Cluster source frame."""

    source: FrameSpace = field(repr=False)
    factor: float = 1.5
    color: Union[int, str] = None
    opacity: float = None
    frame: FrameSpace = field(init=False)

    def __post_init__(self):
        """Apply clustering algorithum."""
        self.frame = self.cluster()

    def cluster(self):
        """Cluster shield panels."""
        points = self.source.loc[:, ['x', 'y', 'z']].values
        eps = self.factor*self.source.dt.max()
        cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=1).fit(points)
        frame = FrameSpace()
        for i, label in enumerate(set(cluster.labels_)):
            index = label == cluster.labels_
            vtksum = self.merge(index)
            name = f'{vtksum.name.rstrip(digits)}{i}'
            frame += vtksum.to_dict() | dict(name=name)
            frame.loc[name, ['x', 'y', 'z']] = self.centroid(index)
        return frame

    def merge(self, index):
        """Return vtk framespace sum."""
        dataseries = self.source.loc[index, :].iloc[0].copy()
        vtk = vedo.merge(*self.source.loc[index, 'vtk'])
        if self.color is not None:
            vtk.c(self.color)
        if self.opacity is not None:
            vtk.opacity(self.opacity)
        dataseries.loc['vtk'] = vtk
        return dataseries

    def centroid(self, index):
        """Return volume weighted centroid."""
        centroids = self.source.loc[index, ['x', 'y', 'z']].values
        volumes = self.source.loc[index, 'volume'].values.reshape(-1, 1)
        centroid = np.sum(centroids * volumes, axis=0)
        centroid /= np.sum(volumes)
        return centroid

    def plot(self):
        """Plot source and clustered vtk framesets."""
        vedo.show(vedo.merge(*self.source.vtk),
                  vedo.merge(*self.frame.vtk).opacity(0.75).c('b'))


@dataclass
class ShieldCluster(ShieldDir):
    """Manage clustered ferritic insert dataset."""

    file: str = 'IWS_CFM'
    frame: FrameSpace = field(init=False, default_factory=FrameSpace)

    def __post_init__(self):
        """Load dataset."""
        super().__post_init__()
        self.frame = self.load()

    def load(self):
        """Load clustered shield frameset."""
        try:
            return FrameSpace().load(self.cdf_file)
        except (FileNotFoundError, OSError):
            return self._build()

    def rebuild(self):
        """Rebuild clustered frameset."""
        self.frame = self._build()

    def _build(self):
        """Build clustered frameset."""
        frame = FrameSpace()
        shield = ShieldSet()  # load source dataset
        parts = shield.frame.part.unique()
        frames = []
        tick = clock(len(parts), header='clustering shield set')
        for i, part in enumerate(parts):
            frames.append(Cluster(shield.frame.loc[part, :],
                                  color=i, opacity=1).frame)
            tick.tock()
        frame.concatenate(*frames)
        frame.store(self.cdf_file, mode='w')
        return frame

    def plot(self):
        """Plot vtk clusters."""
        vedo.show(self.frame.vtk)


@dataclass
class Ferritic(FrameSetLoc):
    """Manage ferritic inserts."""

    delta: float = -1
    required: list[str] = field(default_factory=lambda: ['vtk'])
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Fi', 'part': 'fi', 'ferritic': True})

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for ferritic inserts."""

    def insert(self, *args, required=None, iloc=None, **additional):
        """
        Add ferritic volumes to frameset.

        volumes described by x, y, z centroid and volume.
        Each frame is meshed based on delta.

        Parameters
        ----------
        vtk : Union[str, vedo.Mesh]
            Shield filename or vtk volume.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        '''
        if isinstance(file := args[0], str):  # load frameset from file
            frame = ShieldSet(file).frame
        if isinstance(args[0], FrameSpace):  # unpack frame
            args = frame
            required = frame.columns
        '''
        print(self.required)
        required = self.required

        self.attrs = additional
        with self.insert_required(required):
            print(self.frame.metaframe.required)
            index = self.frame.insert(*args, iloc=iloc, **self.attrs)

        '''
        frame = self.frame.loc[index, :]
        subframe = []
        for i, name in enumerate(index):
            data = frame.iloc[i].to_dict()
            data |= {'label': name, 'frame': name, 'delim': '_', 'link': True}
            subframe.append(self.subframe.assemble(
                shellgrid.subframe[i], **data, **self.subattrs))
        self.subframe.concatenate(*subframe)
        '''


if __name__ == '__main__':

    shield = ShieldSet('IWS_CFM')
    shield.frame.polyplot()
