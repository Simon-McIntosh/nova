"""Manage ILIS cad data as input to Baysean assembly studies."""
from dataclasses import dataclass, field
import os

import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy.spatial.transform import Rotation
import xarray

from nova.definitions import root_dir
from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducialdata import FiducialData


class Surface(pv.PolyData):
    """Create single surface polydata mesh."""

    def __init__(self, points):
        faces = [[len(points), *range(len(points))]]
        super().__init__(points, faces)


@dataclass
class ILIS:
    """Manage ILIS plane definitions."""

    points: npt.ArrayLike
    norm: npt.ArrayLike = field(init=False)
    mesh: pv.PolyData = field(init=False)

    def __post_init__(self):
        """Generate point and normal dataset."""
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points)
        self.points = self.points.copy()
        self.assert_planar()
        self.norm = self.normal(self.points)
        self.mesh = Surface(self.points)

    @staticmethod
    def normal(points):
        """Calculate surface normal."""
        norm = np.cross(points[1] - points[0], points[2] - points[0])
        norm /= np.linalg.norm(norm)
        return norm

    def assert_planar(self):
        """Assert that input points are planar."""
        if len(self.points) == 3:
            return
        assert np.allclose(self.normal(self.points[:3]), self.normal(self.points[1:]))

    def normal_offset(self, delta):
        """Offset by delta in direction of normal."""
        self.__init__(self.points + delta * self.norm)

    def vector_offset(self, delta, vector):
        """Offset by delta in direction of vector."""
        self.__init__(self.points + delta * vector)

    @property
    def pdot(self):
        """Return point[0] @ norm."""
        return self.points[0] @ self.norm

    def rotate_z(self, angle):
        """Rotate ilis plane by angle about z-axis."""
        rotation = Rotation.from_euler("Z", angle, degrees=True)
        self.__init__(rotation.apply(self.points))


@dataclass
class InnerLeg:
    """Manage inner leg TF coil case cad."""

    point_size: float = 6
    mesh: pv.PolyData = field(init=False)
    points: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Load inner leg geometry from cad."""
        ansys = AnsysPost("TFCgapsG10", "k0", "CASE_IL")
        ansys.select(8)
        self.mesh = ansys.mesh

    def store(self, point):
        """Store picked mesh points."""
        self.points.append(list(point))

    def pick_points(self):
        """Pick points interactivly from mesh."""
        self.points = []
        plotter = pv.Plotter()
        plotter.enable_point_picking(callback=self.store, point_size=self.point_size)
        plotter.add_mesh(self.mesh)
        plotter.add_axes()
        plotter.show(window_size=(600, 600))


@dataclass
class OuterLeg:
    """Manage outer leg TF coil case cad."""

    mesh: pv.PolyData = field(init=False)

    def __post_init__(self):
        """Load outer leg geometry from Ansys."""
        ansys = AnsysPost("TFCgapsG10", "k0", "CASE_OL")
        ansys.select(9)
        self.mesh = ansys.mesh


@dataclass
class Case:
    """Manage TF coil case cad."""

    mesh: pv.PolyData = field(init=False)

    def __post_init__(self):
        """Generate inner and outer leg TF coil mesh."""
        self.mesh = InnerLeg().mesh
        self.mesh += OuterLeg().mesh

    def rotate_z(self, angle):
        """Rotate geometry by angle about z-axis."""
        self.mesh = self.mesh.rotate_z(angle, inplace=False)


@dataclass
class Coil:
    """Manage ILIS coil data."""

    ilis: list[ILIS]
    gap: float = 0.006
    thickness: float = 0.002
    case: Case = field(default_factory=Case)
    origin: npt.ArrayLike = field(init=False)

    def __post_init__(self):
        """Calculate ILIS plane origin."""
        self.assert_wedge()
        self.calculate_origin()
        self.correct_radial_offset()
        self.add_plate_thickness()

    def assert_wedge(self):
        """Assert wedge angle == pi/9."""
        assert np.isclose(
            np.pi - np.arccos(np.dot(self.ilis[0].norm, self.ilis[1].norm)), np.pi / 9
        )

    def calculate_origin(self):
        """Calculate ILIS plane origin."""
        point_matrix = np.zeros((5, 5))
        point_matrix[:3, :3] = 2 * np.identity(3)
        point_matrix[3, :3] = self.ilis[0].norm
        point_matrix[:3, 3] = self.ilis[0].norm
        point_matrix[4, :3] = self.ilis[1].norm
        point_matrix[:3, 4] = self.ilis[1].norm
        self.origin = np.linalg.solve(
            point_matrix, [0, 0, 0, self.ilis[0].pdot, self.ilis[1].pdot]
        )[:3]

    @property
    def radius(self):
        """Return origin radius."""
        return np.linalg.norm(self.origin)

    def correct_radial_offset(self):
        """Offset coil to maintain target inter-coil gap."""
        radius = self.radius
        offset = self.gap / 2 / np.sin(np.pi / 18) - radius
        for i in range(2):
            self.ilis[i].vector_offset(offset, self.origin / radius)
        self.case.mesh.translate(offset * self.origin / radius, inplace=True)
        self.calculate_origin()
        assert np.isclose(self.radius * np.sin(np.pi / 18), self.gap / 2)

    def add_plate_thickness(self):
        """Offset ilis plate from TF coil by thickness."""
        for ilis in self.ilis:
            ilis.normal_offset(self.thickness)

    def rotate_z(self, angle, case=True):
        """Rotate geometry by angle about z-axis."""
        if case:
            self.case.rotate_z(angle)
        for ilis in self.ilis:
            ilis.rotate_z(angle)
        self.origin = Rotation.from_euler("Z", angle, degrees=True).apply(self.origin)
        return self

    def plot(self, plotter=None, show=True, case=True):
        """Plot inner leg case and ilis surfaces."""
        if plotter is None:
            plotter = pv.Plotter()
        if case:
            plotter.add_mesh(self.case.mesh)
        plotter.add_mesh(self.ilis[0].mesh, color="r")
        plotter.add_mesh(self.ilis[1].mesh, color="b")
        plotter.add_mesh(
            pv.Line([*self.origin[:2], -5], [*self.origin[:2], 5]), color="w"
        )
        plotter.add_axes()
        if show:
            plotter.show(window_size=(600, 600))
        return plotter


@dataclass
class Cage:
    """Build ILIS cage."""

    ilis_a: npt.ArrayLike
    ilis_b: npt.ArrayLike
    gap: float = 0.006
    thickness: float = 0.002
    mesh: pv.PolyData = field(init=False, default_factory=pv.PolyData)

    def __post_init__(self):
        """Pattern referance coil."""
        self.coil = Coil(
            [ILIS(self.ilis_a), ILIS(self.ilis_b)], self.gap, self.thickness
        )
        self.data = xarray.Dataset(
            coords=dict(
                coil=range(1, 19),
                side=["a", "b"],
                corner=range(4),
                coordinate=["x", "y", "z"],
            )
        )
        self.data = self.data.assign_coords(
            azimuth=(
                "coil",
                [
                    FiducialData.location.index(coil) * np.pi / 9
                    for coil in self.data.coil
                ],
            )
        )
        self.data = self.data.sortby("azimuth")

        self.data["ilis"] = (
            ("coil", "side", "corner", "coordinate"),
            np.zeros((18, 2, 4, 3)),
        )
        self.data["point"] = (("coil", "side", "coordinate"), np.zeros((18, 2, 3)))
        self.data["normal"] = (("coil", "side", "coordinate"), np.zeros((18, 2, 3)))
        for i in range(18):
            for j in range(2):
                self.data["ilis"][i, j] = self.coil.ilis[j].points
                self.data["point"][i, j] = self.coil.ilis[j].points.mean(axis=0)
                self.data["normal"][i, j] = self.coil.ilis[j].norm
            self.coil.rotate_z(20, case=False)
        self.data.attrs = dict(
            unit="m",
            ilis_thickness=self.coil.thickness,
            ilis_gap=self.coil.gap - 2 * self.coil.thickness,
            description="Nominal ILIS surfaces extracted "
            "from CAD. Coils are ordered by azimuth and "
            "indexed consistent with the ITER assembly "
            "scheme. Each coil is represented by a pair of "
            "ILIS planes ordered by azimuth and labeled "
            '"a"", ""b". The corners of '
            "each ILIS plane are provided by data variable "
            '"ilis", the center of each plane by data '
            'variable "point" and the normal of each plane '
            'by the data variable "normal".',
        )
        self.data.to_netcdf(self.file)

    @property
    def file(self):
        """Return data filename."""
        return os.path.join(root_dir, "input/ITER", "ilis_planes.nc")

    def plot_corners(self, plotter=None, show=True):
        """Plot data."""
        if plotter is None:
            plotter = pv.Plotter()
        plotter.add_mesh(
            pv.PolyData(self.data.ilis[:, 0].data.reshape(-1, 3)), color="r"
        )
        plotter.add_mesh(
            pv.PolyData(self.data.ilis[:, 1].data.reshape(-1, 3)), color="b"
        )
        if show:
            plotter.show(window_size=(600, 600))
        return plotter

    def plot_mesh(self, plotter=None, show=True):
        """Plot ILIS surfaces and TF coil case mesh."""
        if plotter is None:
            plotter = pv.Plotter()
        coil = Coil([ILIS(self.ilis_a), ILIS(self.ilis_b)], self.gap, self.thickness)
        for i in range(18):
            coil.rotate_z(20)
            coil.plot(plotter, show=False, case=False)
        plotter.add_mesh(coil.case.mesh)
        if show:
            plotter.show(window_size=(600, 600))
        return plotter


if __name__ == "__main__":
    ilis_a = [
        [3.1102735026974, -0.544363429461, 4.3580624691689],
        [2.2785593144684, -0.397709777838, 4.7462638399745],
        [2.2760361378485, -0.397264873723, -4.679000536587],
        [3.2061785302121, -0.561274073397, -4.577528574906],
    ]
    ilis_b = [
        [2.2783856668339473, 0.3986945825098288, 4.7462638399745],
        [3.1100998550629804, 0.5453482341331086, 4.358062469168898],
        [3.2060048825775693, 0.5622588780695716, -4.577528574906],
        [2.275862490214159, 0.3982496783946659, -4.679000536586998],
    ]

    cage = Cage(ilis_a, ilis_b, gap=7e-3)
    print(cage.coil.radius)
    plotter = cage.plot_mesh(show=False)
    cage.plot_corners(plotter)

    ilis = np.array(ilis_a)
    points = np.array([np.linalg.norm(ilis[:, :2], axis=1), ilis[:, -1]]).T

    gap_points = np.zeros((3, 2))
    gap_points[:2] = points[1:3]
    gap_points[-1] = np.mean([points[0], points[-1]], axis=0)

    import pandas

    point_data = pandas.DataFrame(
        1e3 * gap_points.T, columns=["A1", "C1", "B2"], index=["radius", "height"]
    )
