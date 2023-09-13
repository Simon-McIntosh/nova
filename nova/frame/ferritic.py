"""Manage ferromagnetic insert geometry."""
from dataclasses import dataclass, field
import os
from string import digits
from tqdm import tqdm
from typing import ClassVar, Union

import numpy as np
import sklearn.cluster
import vedo

from nova.definitions import root_dir
from nova.frame.coilsetattrs import CoilSetAttrs
from nova.frame.framespace import FrameSpace


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
            raise IndexError(f"sector {self.sector} not in {self.avalible}")


@dataclass
class ShieldDir:
    """Manage shield filepath."""

    file: str = "Fi"
    data_dir: str | None = None
    source_dir: str = "/mnt/share/shield"  # //io-ws-ccstore1/ANSYS_Data/mcintos

    def __post_init__(self):
        """Define file paths."""
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if self.data_dir is None:
            self.data_dir = os.path.join(root_dir, "input/ITER/shield")

    @property
    def cdf_file(self):
        """Return netCDF filename."""
        return os.path.join(self.data_dir, f"{self.file}.nc")


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
    def cdf_file(self):
        """Return sector cdf filename as _{file}.cdf."""
        path, file = os.path.split(super().cdf_file)
        return os.path.join(path, f"_{file}")

    @property
    def data(self):
        """Return mesh and frame data."""
        return self.mesh, self.frame

    @property
    def frame_metadata(self):
        """Return frame metadata."""
        return {
            "label": f"fi{self.sector}",
            "segment": "panel",
            "delim": "_",
        }

    def frame_data(self, vtk: vedo.Mesh):
        """Return frame data."""
        return {"vtk": vtk, "part": f"fi{self.sector}"}

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.data_dir, f"{self.file}_S{self.sector}.vtk")

    @property
    def stl_file(self):
        """Return full stl filename."""
        filename = f"{self.file}_S{self.sector}.stl"
        if os.path.isfile(file := os.path.join(self.data_dir, filename)):
            return file
        if os.path.isfile(file := os.path.join(self.source_dir, filename)):
            return file
        raise FileNotFoundError(
            f"file {filename} not found in: \n{self.data_dir} \n{self.source_dir}"
        )

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
            return FrameSpace().load(self.cdf_file, f"S{self.sector}")
        except (FileNotFoundError, OSError):
            return self.build_frame()

    def build_frame(self):
        """Return dataframe read from vtk file."""
        parts = self.mesh.split(10000)
        frame = FrameSpace(**self.frame_metadata)
        for i, part in enumerate(tqdm(parts, "loading decimated convex hulls")):
            frame += self.frame_data(part.c(i))
        self.store_frame(frame)
        return frame

    def store_frame(self, frame):
        """Write frame to netCDF file."""
        mode = "a" if os.path.isfile(self.cdf_file) else "w"
        frame.store(self.cdf_file, f"S{self.sector}", mode=mode, vtk=True)

    def plot(self):
        """Plot vtk mesh."""
        vedo.show(self.frame.vtk)


@dataclass
class ShieldSector(ShieldCad):
    """Manage shield sector rotations."""

    avalible: ClassVar[list[int]] = [*range(1, 10)]
    base_sector: ClassVar[int] = 6

    def build_mesh(self):
        """Extend ShieldCad build sector mesh via base rotation."""
        if self.sector in ShieldCad.avalible:
            return super().build_mesh()
        return self.rotate("mesh")

    def build_frame(self):
        """Extend ShieldCad build sector frame via base rotation."""
        if self.sector in ShieldCad.avalible:
            return super().build_frame()
        return self.rotate("frame")

    def rotate(self, attribute: str):
        """Perform base rotation."""
        if hasattr(self, attribute):
            return getattr(self, attribute)
        self._rotate()
        return getattr(self, attribute)

    def _rotate(self):
        """Generate mesh and frame data via rotation from base."""
        degrees = 40 * (self.sector - self.base_sector)
        base = ShieldCad(self.base_sector, file=self.file)
        self.mesh = self.rotate_mesh(base.mesh, degrees)
        self.frame = self.rotate_frame(base.frame, degrees)

    def rotate_mesh(self, base_mesh: vedo.Mesh, degrees: float):
        """Return rotated mesh and save output to vtk file."""
        mesh = base_mesh.clone().rotate(degrees, axis=(0, 0, 1), point=(0, 0, 0))
        self.store_mesh(mesh)
        return mesh

    def rotate_frame(self, base_frame: FrameSpace, degrees: float):
        """Rotate base frame and save output to netCDF file."""
        frame = FrameSpace(**self.frame_metadata)
        for vtk in tqdm(
            base_frame.loc[:, "vtk"],
            f"rotating baseframe {self.base_sector} " f"to {self.sector}",
        ):
            vtk = vtk.clone().rotate(degrees, axis=(0, 0, 1), point=(0, 0, 0))
            frame += self.frame_data(vtk)
        self.store_frame(frame)
        return frame


@dataclass
class ShieldSet(ShieldDir):
    """Manage complete ferritic insert dataset."""

    file: str = "Fi"
    mesh: vedo.Mesh = field(init=False, repr=False)
    frame: FrameSpace = field(init=False, repr=False)
    avalible: ClassVar[list[int]] = [*range(1, 10)]

    def __post_init__(self):
        """Load shieldset."""
        super().__post_init__()
        self.mesh = self.load_mesh()
        self.frame = self.load_frame()

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.data_dir, f"{self.file}.vtk")

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
        mesh = []
        for i in tqdm(self.avalible, "building ferritic insert mesh"):
            sector = ShieldSector(i, file=self.file)
            mesh.append(sector.mesh)
        mesh = vedo.merge(mesh)
        mesh.write(self.vtk_file)
        return mesh

    def build_frame(self):
        """Build complete ferritic sheildset dataframe."""
        frame = []
        for i in tqdm(self.avalible, "building ferritic insert dataset"):
            sector = ShieldSector(i, file=self.file)
            frame.append(sector.frame)
        frame = FrameSpace().concatenate(*frame)
        frame.store(self.cdf_file, mode="w", vtk=True)
        return frame

    def plot_mesh(self):
        """Plot vtk mesh."""
        vedo.show(self.mesh, *self.frame.vtk, axes=3)

    def plot_points(self):
        """Plot centroid point cloud."""
        vedo.show(vedo.Mesh(self.frame.loc[:, ["x", "y", "z"]].values))


@dataclass
class Cluster:
    """Cluster source frame."""

    source: FrameSpace = field(repr=False)
    factor: float = 1.5
    color: Union[int, str] | None = None
    opacity: float | None = None
    frame: FrameSpace = field(init=False)

    def __post_init__(self):
        """Apply clustering algorithum."""
        self.frame = self.cluster()

    def cluster(self):
        """Cluster shield panels."""
        points = self.source.loc[:, ["x", "y", "z"]].values
        eps = self.factor * self.source.dt.max()
        cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=1).fit(points)
        frame = FrameSpace(segment="panel")
        for i, label in enumerate(set(cluster.labels_)):
            index = label == cluster.labels_
            vtksum = self.merge(index)
            name = f"{vtksum.name.rstrip(digits)}{i}"
            frame += vtksum.to_dict() | {"name": name}
            frame.loc[name, ["x", "y", "z"]] = self.centroid(index)
        return frame

    def merge(self, index):
        """Return vtk framespace sum."""
        dataseries = self.source.loc[index, :].iloc[0].copy()
        vtk = vedo.merge(*self.source.loc[index, "vtk"])
        if self.color is not None:
            vtk.c(self.color)
        if self.opacity is not None:
            vtk.opacity(self.opacity)
        dataseries.loc["vtk"] = vtk
        return dataseries

    def centroid(self, index):
        """Return volume weighted centroid."""
        centroids = self.source.loc[index, ["x", "y", "z"]].values
        volumes = self.source.loc[index, "volume"].values.reshape(-1, 1)
        centroid = np.sum(centroids * volumes, axis=0)
        centroid /= np.sum(volumes)
        return centroid

    def plot(self):
        """Plot source and clustered vtk framesets."""
        vedo.show(
            vedo.merge(*self.source.vtk),
            vedo.merge(*self.frame.vtk).opacity(0.75).c("b"),
        )


@dataclass
class ShieldCluster(ShieldDir):
    """Manage clustered ferritic insert dataset."""

    file: str = "Fi"
    frame: FrameSpace = field(init=False, default_factory=FrameSpace)

    def __post_init__(self):
        """Load dataset."""
        super().__post_init__()
        self.frame = self.load()

    @property
    def cdf_file(self):
        """Return clustered cfd filename."""
        file, ext = os.path.splitext(super().cdf_file)
        return f"{file}_cl{ext}"

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
        frame = FrameSpace(segment="panel")
        shield = ShieldSet(self.file)  # load source dataset
        parts = shield.frame.part.unique()
        frames = []
        for i, part in enumerate(tqdm(parts, "clustering shield set")):
            index = shield.frame.part == part
            frames.append(Cluster(shield.frame.loc[index, :], color=i, opacity=1).frame)
        frame.concatenate(*frames)
        frame.store(self.cdf_file, mode="w", vtk=True)
        return frame

    def plot(self):
        """Plot vtk clusters."""
        vedo.show(self.frame.vtk)


@dataclass
class FerriticBase(CoilSetAttrs):
    """Ferritic insert baseclass."""

    delta: float = -1
    name: str | None = None
    default: dict = field(
        init=False,
        default_factory=lambda: {
            "label": "Fi",
            "part": "fi",
            "ferritic": True,
            "active": False,
            "segment": "vtk",
            "link": "",
            "factor": False,
            "ref": 0,
            "subref": 0,
        },
    )

    @property
    def attrs(self):
        """Extend CoilSet attrs."""
        return CoilSetAttrs.attrs.fget(self)

    @attrs.setter
    def attrs(self, attrs):
        CoilSetAttrs.attrs.fset(self, attrs)
        self.attrs.pop("vtk", None)

    def set_conditional_attributes(self):
        """Set conditional attributes. Not currently required for ferritic class."""

    def insert(self, vtk, iloc=None, **additional):
        """
        Add ferritic volumes to frameset.

        volumes described by vtk instance .
        frame properties calculated by Vtkgeo (x, y, z centroid and volume)

        Parameters
        ----------
        vtk : Union[vedo.Mesh, list[vedo.Mesh]]
            vtk volumes.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        self.attrs = additional
        with self.insert_required("vtk"):
            name = self.frame.build_index(1, **self.attrs)[0]
            self.attrs.pop("name", None)
            self.attrs |= {
                "frame": name,
                "label": name,
                "delim": "_",
            }
            # insert subframes
            index = self.subframe.insert(vtk, iloc=iloc, **self.attrs)
            subframe = self.subframe.loc[index, :]
            # insert frame
            self.attrs |= subframe.iloc[0].to_dict()
            self.attrs |= {"delim": "", "name": name, "segment": "vtk"}
            self.attrs.pop("poly", None)
            vtk = vedo.merge(*subframe.vtk)
            self.frame.insert(vtk, iloc=iloc, **self.attrs)


@dataclass
class Ferritic(FerriticBase):
    """Manage ferritic inserts."""

    cluster: bool = True

    def insert(self, vtk, iloc=None, **additional):
        """
        Add ferritic volumes to frameset.

        volumes described by vtk instance.
        frame properties calculated by Vtkgeo (x, y, z centroid and volume)

        Parameters
        ----------
        vtk : Union[str, DataFrame, vedo.Mesh, list[vedo.Mesh]]
            Shield filename or vtk volumes.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        if isinstance(file := vtk, str):  # load frameset from file
            frame = self.load_frame(file)
            return self.insert_frame(frame, **additional)
        super().insert(vtk, iloc=iloc, **additional)

    def _update_label(self, additional):
        if "label" in additional:
            self.attrs["label"] = additional["label"]
            self.attrs.pop("name", None)

    def insert_frame(
        self, frame, multiframe=True, iloc=None, segment="vtk", **additional
    ):
        """Insert vtk objects from frame."""
        if not multiframe:
            self.attrs = additional | frame.iloc[0].to_dict()
            self._update_label(additional)
            return super().insert(frame, iloc=iloc, **self.attrs)
        for part in frame.part.unique():
            index = part == frame.part
            _frame = frame.iloc[np.argmax(index)]
            self.attrs = additional | _frame.to_dict()
            self.attrs["name"] = _frame.get("frame", _frame["part"])
            self._update_label(additional)
            super().insert(frame.loc[index, :], iloc=iloc, **self.attrs)

    def load_frame(self, file: str):
        """Load frame from file."""
        if os.path.isfile(file):
            return FrameSpace().load(file)
        if self.cluster:
            return ShieldCluster(file).frame
        return ShieldSet(file).frame


if __name__ == "__main__":
    # cad = ShieldCad(2)
    # sector = ShieldSector(9)
    # sector.mesh.show(new=True)
    # shield = ShieldSet()
    # cluster = ShieldCluster()

    from nova.frame.coilset import CoilSet

    coilset = CoilSet()
    coilset.ferritic.insert("Fi")
    coilset.plot()

    """
    cad = ShieldCad(2)
    print(sum([ShieldCad(i).frame.volume.sum() for i in range(1, 10)]))
    base.frame = base.build_frame()
    print(base.frame.volume.sum())
    print(base.frame.volume.sum())
    # shield = ShieldSet()
    # shield.frame = shield.build_frame()
    # print(shield.frame.volume.sum())

    [ShieldCad(i).mesh.volume() for i in [2, 3, 4, 6]]
    # [ShieldCad(i).mesh.volume() for i in [2, 3, 4, 6]]
    # shield = ShieldSet()

    shield = ShieldCluster()
    # print(shield.frame.volume.sum())
    # vedo.show(*shield.frame.vtk, new=True)
    """

    """
    mesh = []
    for i in range(1, 5):
        mesh.append(ShieldSector(i).mesh.rotate_z(-i*40).c(i).opacity(0.5))
    vedo.show(mesh)
    """
