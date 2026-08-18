"""Generate grid and solution methods for hexagonal plasma filaments."""

from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module

import numpy as np
from shapely.geometry.linestring import LineString

from nova.biot.biotframe import Target
from nova.biot.error import PlasmaTopologyError
from nova.biot.grid import BaseGrid
from nova.biot.solve import Solve
from nova.frame.error import GridError
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop

from nova.frame.plasmaloc import PlasmaLoc


@dataclass
class PlasmaGrid(BaseGrid, PlasmaLoc):
    """Compute interaction across hexagonal grid."""

    attrs: list[str] = field(
        default_factory=lambda: [
            "Br",
            "BrR",
            "BrZ",
            "Bz",
            "BzR",
            "BzZ",
            "Psi",
            "PsiR",
            "PsiZ",
        ]
    )
    levels: int | list[float] | np.ndarray = 21
    sample_data: object | None = field(init=False, repr=False, default=None)
    _sampling_vertices: np.ndarray | None = field(init=False, repr=False, default=None)
    _sample_coordinates: np.ndarray | None = field(init=False, repr=False, default=None)
    _cell_sample_nodes: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize psi axis and psi x versions."""
        super().__post_init__()
        self.version["psi_axis"] = None
        self.version["psi_x"] = None

    def __getitem__(self, attr):
        """Implement dict-like access to plasmagrid attributes."""
        match attr:
            case "x_point":
                return self.x_points[self._x_point_index()]
            case "x_psi":
                return self.x_psi[self._x_point_index()]
            case "o_point":
                return self.o_points[self._o_point_index()]
            case "o_psi":
                return self.o_psi[self._o_point_index()]
        if hasattr(self, "__getitem__"):
            return super().__getitem__(attr)

    def _x_point_index(self):
        """Return x-point index for primary plasma separatrix."""
        match self.x_point_number:
            case 0:
                raise PlasmaTopologyError("no x-points within first wall")
        x_psi = self.x_psi.copy()
        if self.o_point_number > 0:
            x_psi -= self.o_psi[0]
        return np.nanargmax(self.polarity * (x_psi))

    def _o_point_index(self):
        """Return plasma o-point index."""
        match self.o_point_number:
            case 1:
                return 0
            case 0:
                raise PlasmaTopologyError("no o-points found within first wall")
            case _:
                return 0
                raise PlasmaTopologyError(
                    "multiple o-points found within first wall {self.data_o}"
                )

    def solve(self):
        """Solve Biot interaction across plasma grid."""
        if self.sloc["plasma"].sum() == 0:
            raise GridError("plasma")
        target = Target(
            {attr: self.aloc["plasma", attr] for attr in ["x", "z", "poly"]}
        )
        wall = self.ALoc["plasma", "poly"][0].poly.boundary
        self.data = Solve(
            self.subframe,
            target,
            reduce=[True, False],
            attrs=self.attrs,
            name=self.name,
        ).data
        self.tessellate(target, wall)
        self._build_direct_samples(target)
        super().post_solve()

    @staticmethod
    def _ordered_polygon_vertices(polygon, centre) -> np.ndarray:
        """Return one authored polygon's exterior in angular order."""
        geometry = polygon.poly if hasattr(polygon, "poly") else polygon
        vertices = np.asarray(geometry.exterior.coords, dtype=np.float64)[:-1, :2]
        offset = vertices - centre
        return vertices[np.argsort(np.arctan2(offset[:, 1], offset[:, 0]))]

    def _build_direct_samples(self, target: Target) -> None:
        """Bank the authoritative pre-clip hex vertices and their coupling rows."""
        centres = np.c_[np.asarray(target.x), np.asarray(target.z)]
        sections = np.asarray(target["section"], dtype=object).astype(str)
        full = np.flatnonzero(sections == "hexagon")
        if len(full) == 0:
            raise ValueError(
                "a hex plasma grid needs at least one complete generator cell"
            )
        offsets = np.stack(
            [
                self._ordered_polygon_vertices(target.poly[cell], centres[cell])
                - centres[cell]
                for cell in full
            ]
        )
        if offsets.shape[1:] != (6, 2):
            raise ValueError("complete hex generator cells must carry six vertices")
        canonical = np.median(offsets, axis=0)
        scale = max(
            float(np.max(np.abs(centres))),
            float(np.max(np.abs(canonical))),
            1.0,
        )
        tolerance = 128.0 * np.finfo(np.float64).eps * scale
        deviation = float(np.max(np.abs(offsets - canonical)))
        if deviation > tolerance:
            raise ValueError(
                "complete hex generator offsets do not collapse at round-off: "
                f"{deviation:.17g} m exceeds {tolerance:.17g} m"
            )
        vertices = centres[:, None, :] + canonical[None, :, :]
        flat = vertices.reshape(-1, 2)
        lookup: dict[tuple[int, int], int] = {}
        coordinates = []
        inverse = np.empty(len(flat), dtype=np.intp)
        for index, vertex in enumerate(flat):
            key = tuple(np.rint(vertex / tolerance).astype(np.int64))
            if key not in lookup:
                lookup[key] = len(coordinates)
                coordinates.append(vertex)
            inverse[index] = lookup[key]
        coordinates = np.asarray(coordinates)
        self._sampling_vertices = vertices
        self._sample_coordinates = coordinates
        self._cell_sample_nodes = inverse.reshape(len(centres), 6)
        sample_target = Target(
            {"x": coordinates[:, 0], "z": coordinates[:, 1]}, label="PlasmaSample"
        )
        self.sample_data = Solve(
            self.subframe,
            sample_target,
            reduce=[True, False],
            attrs=self.attrs,
            name=self.name,
        ).data
        for name in ("target", "sample_target"):
            self.__dict__.pop(name, None)

    @property
    def sampling_vertices(self) -> np.ndarray:
        """Return the authoritative six pre-clip vertices of every plasma cell."""
        if self._sampling_vertices is None:
            raise AttributeError("solve the plasma grid before reading sample vertices")
        return self._sampling_vertices

    @property
    def sample_coordinates(self) -> np.ndarray:
        """Return the unique direct-target coordinates used by the vertex gather."""
        if self._sample_coordinates is None:
            raise AttributeError("solve the plasma grid before reading sample targets")
        return self._sample_coordinates

    @property
    def cell_sample_nodes(self) -> np.ndarray:
        """Return each cell's six indices into the unique sample coordinates."""
        if self._cell_sample_nodes is None:
            raise AttributeError("solve the plasma grid before reading sample targets")
        return self._cell_sample_nodes

    @cached_property
    def target(self):
        """Return the centre-target coupling carried by this plasma grid."""
        import jax.numpy as jnp  # noqa: PLC0415

        from nova.biot.null import Null2D  # noqa: PLC0415
        from nova.biot.target import FluxTarget  # noqa: PLC0415

        null = Null2D.from_coordinates(
            np.c_[self.data.x, self.data.z], np.asarray(self.data.stencil), maxsize=5
        )
        return FluxTarget(
            source_target=jnp.asarray(self.data["Psi"])[:, :-1],
            plasma_target=jnp.asarray(self.data["Psi_"]),
            null=null,
            plasma_target_r=jnp.asarray(self.data["PsiR_"]),
            plasma_target_z=jnp.asarray(self.data["PsiZ_"]),
        )

    @cached_property
    def sample_target(self):
        """Return the direct pre-clip vertex coupling target."""
        if self.sample_data is None or self._sample_coordinates is None:
            raise AttributeError("solve the plasma grid before reading sample targets")
        import jax.numpy as jnp  # noqa: PLC0415

        from nova.biot.null import Null1D  # noqa: PLC0415
        from nova.biot.target import FluxTarget  # noqa: PLC0415

        return FluxTarget(
            source_target=jnp.asarray(self.sample_data["Psi"])[:, :-1],
            plasma_target=jnp.asarray(self.sample_data["Psi_"]),
            null=Null1D(jnp.asarray(self._sample_coordinates)),
            plasma_target_r=jnp.asarray(self.sample_data["PsiR_"]),
            plasma_target_z=jnp.asarray(self.sample_data["PsiZ_"]),
        )

    @staticmethod
    def loop_neighbour_vertices(points, neighbor_vertices, boundary_vertices):
        """Calculate 6-point ordered loop vertex indices."""
        point_number = len(points)
        neighbours = np.full((point_number, 6), -1)
        for i in range(len(points)):
            if i in boundary_vertices:
                continue
            center_point = points[i, :]
            slice_index = slice(neighbor_vertices[0][i], neighbor_vertices[0][i + 1])
            neighbour_index = neighbor_vertices[1][slice_index]
            if len(neighbour_index) != 6:
                continue
            delta = points[neighbour_index] - center_point
            angle = np.arctan2(delta[:, 1], delta[:, 0])
            neighbours[i] = neighbour_index[np.argsort(angle)[::-1]]
        mask = neighbours[:, 0] != -1
        stencil_index = np.arange(point_number)[mask]
        stencil = np.append(
            np.arange(point_number)[mask].reshape(-1, 1), neighbours[mask], axis=1
        )
        return stencil, stencil_index

    def tessellate(self, target: Target, wall: LineString):
        """Tesselate hexagonal mesh, compute 6-point neighbour loops."""
        points = np.c_[target.x, target.z]
        tri = import_module("scipy.spatial").Delaunay(points)
        neighbor_vertices = tri.vertex_neighbor_vertices
        boundary_vertices = np.array(
            [
                i
                for i, polygon in enumerate(target.poly)
                if polygon.poly.intersects(wall)
            ]
        )
        centroids = np.array(
            [np.mean(points[simplex], axis=0) for simplex in tri.simplices]
        )
        inside = PointLoop(centroids).update(np.array(wall.xy).T)
        triangles = tri.simplices[inside]
        stencil, stencil_index = self.loop_neighbour_vertices(
            points, neighbor_vertices, boundary_vertices
        )
        self.data.coords["x"] = points[:, 0]
        self.data.coords["z"] = points[:, 1]
        self.data.coords["stencil_index"] = stencil_index
        self.data["triangles"] = ("tri_index", "tri_vertex"), triangles
        self.data["stencil"] = ("stencil_index", "stencil_vertex"), stencil

    def psi_mask(self, psi):
        """Return plasma filament psi-mask."""
        if self.polarity > 0:
            return self.psi >= psi
        return self.psi < psi

    def x_mask(self, z_plasma: np.ndarray):
        """Return plasma filament x-mask."""
        mask = np.ones(len(z_plasma), dtype=bool)
        if self.x_point_number == 0 or self.o_point_number == 0:
            return mask
        o_point = self.o_points[0]
        for x_point in self.x_points[: self.x_point_number]:
            if x_point[1] < o_point[1]:
                mask &= z_plasma > x_point[1]
            else:
                mask &= z_plasma < x_point[1]
        return mask

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        if self.saloc["plasma"].sum() == 0:
            raise AttributeError("No plasma filaments found.")
        return PointLoop(np.c_[self.aloc["plasma", "x"], self.aloc["plasma", "z"]])

    def ionize_mask(self, index):
        """Return plasma filament selection mask."""
        match index:
            case int(psi) | float(psi):
                # case psi if isinstance(psi, (int, float, np.ndarray)):
                z_plasma = self.aloc["plasma", "z"]
                mask = self.psi_mask(psi)
                try:
                    return mask & self.x_mask(z_plasma)
                except IndexError:
                    return mask
            case [int(psi) | float(psi), float(z_min)]:
                return self.psi_mask(psi) & self.aloc["plasma", "z"] > z_min
            case [int(psi) | float(psi), float(z_min), float(z_max)]:
                z_plasma = self.aloc["plasma", "z"]
                return self.psi_mask(psi) & z_plasma > z_min & z_plasma < z_max
            case _:
                try:
                    return self.pointloop.update(index)
                except Exception:  # numba.TypingError:
                    index = Polygon(index).boundary
                    return self.pointloop.update(index)

    def _label_format(self, value):
        return f"{1e3 * value: 1.1f}"

    def plot(self, attr="psi", clabel=False, nulls=True, **kwargs):
        """Plot poloidal flux contours."""
        if nulls and hasattr(self, "psi"):
            super().plot(axes=kwargs.get("axes", None))
        kwargs = self.contour_kwargs(**kwargs)
        if kwargs.pop("plot_mesh", False):
            self.axes.triplot(
                self.data.x,
                self.data.z,
                self.data.triangles,
                lw=0.5,
                color="C0",
                alpha=0.2,
            )
        if isinstance(attr, str):
            attr = getattr(self, attr)
        label = kwargs.pop("label", "")
        contour = self.axes.tricontour(
            self.data.x, self.data.z, self.data.triangles, attr, **kwargs
        )
        self.label_contour(label, **kwargs)
        if clabel:
            self.axes.clabel(
                contour,
                contour.levels[::2],
                inline=True,
                fmt=self._label_format,
                fontsize="small",
            )
        return contour.levels
